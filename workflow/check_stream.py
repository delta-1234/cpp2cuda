"""Regression gates for segment tails, batch rollover, symbol scoping and service reuse."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from check import edge_input
from example_paths import EXPECTED_OUTPUT, SAMPLE_TICKS
from scale import ROOT, Worker, file_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, type=Path)
    p.add_argument("--with-triton", action="store_true")
    args = p.parse_args()
    run = Path(tempfile.mkdtemp(prefix="stream-check-",dir=ROOT/"runs"))
    edge = run/"edge.csv"
    edge_input(edge)
    empty = run/"empty.csv"
    many = run/"many.csv"
    with (ROOT/SAMPLE_TICKS).open() as src, many.open("w") as dst:
        header = src.readline()
        empty.write_text(header)
        dst.write(header)
        for row in src:
            stamp,symbol,rest = row.split(",",2)
            for n in range(73):
                dst.write(f"{stamp},{symbol}_{n},{rest}")
    # Keep 8191, 8192 and 8193 eligible rows from the sparse-symbol tape.
    # Row keys identify the same trade in the unfiltered input.
    gold_many = run/"many-ref.csv"
    subprocess.run([args.exe,"legacy",many,gold_many],check=True,stdout=subprocess.DEVNULL)
    wanted = {}
    with gold_many.open() as f:
        next(f)
        for n,line in enumerate(f,1):
            if n in (8191,8192,8193):
                t,s,count = line.split(",")[:3]
                wanted[(t,s,int(count))] = n
            if n>=8193: break
    cuts = []
    trade_counts = {}
    with many.open() as f:
        data = [f.readline()]
        for line in f:
            data.append(line)
            t,s,kind = line.split(",",3)[:3]
            if kind != "T": continue
            trade_counts[s] = trade_counts.get(s,0)+1
            n = wanted.get((t,s,trade_counts[s]))
            if n:
                path = run/f"tail-{n}.csv"
                path.write_text("".join(data))
                cuts.append(path)
            if len(cuts)==3: break
    if len(cuts)!=3: raise ValueError("failed to construct batch-boundary fixtures")
    modes = ["cpu","simt","tile"]+(["triton"] if args.with_triton else [])
    report = {}
    workers = {}
    try:
        for mode in modes: workers[mode] = Worker(args.exe.resolve(),mode,True)
        for src in [ROOT/SAMPLE_TICKS,edge,*cuts,many,empty,edge]:
            gold = run/"cpu-ref.csv"
            subprocess.run([args.exe,"legacy",src,gold],check=True,stdout=subprocess.DEVNULL)
            sha,rows = file_info(gold)
            if src.name=="sample_ticks.csv" and sha!=file_info(ROOT/EXPECTED_OUTPUT)[0]:
                raise ValueError("sample differs from repository fixture")
            for mode,w in workers.items():
                dst = run/f"{mode}.csv"
                _,actual,*_ = w.run(src,dst)
                if actual!=rows or file_info(dst)[0]!=sha: raise ValueError(f"{mode}: {src}")
            report[src.name] = {"forecasts":rows,"sha256":sha,"verified":modes}
    finally:
        for w in workers.values(): w.close()
    (run/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(run/"checks.json")


if __name__=="__main__": main()
