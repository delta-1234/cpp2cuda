"""Matched CSV I/O ablation for the old packed SIMT and in-memory SIMT paths."""
import argparse
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from check import materialize
from scale import ROOT, Worker, file_info


def time_ms(fn):
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns()-start)/1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe",required=True,type=Path)
    p.add_argument("--packed-simt",required=True,type=Path)
    p.add_argument("--cpu",required=True,type=Path)
    p.add_argument("--repeat",type=int,default=5)
    p.add_argument("--core",type=int)
    p.add_argument("inputs",nargs="+",type=Path)
    args = p.parse_args()
    if args.core is not None: os.sched_setaffinity(0,{args.core})
    run = Path(tempfile.mkdtemp(prefix="io-",dir=ROOT/"runs"))
    report = []
    rng = random.Random(4092)
    for src in args.inputs:
        windows,keys,rates,result = [run/name for name in ("windows.bin","keys.csv","rates.bin","result.csv")]
        ref = run/"reference.csv"
        with ref.open("wb") as f:
            subprocess.run([args.cpu,src],stdout=f,check=True)
        expected,rows = file_info(ref)
        samples = []
        for _ in range(args.repeat):
            trial = {}
            # Alternate tmp-file vs /dev/null packing to reduce ordering bias.
            routes = [("pack_to_files_ms",windows,keys),("pack_to_null_ms",os.devnull,os.devnull)]
            rng.shuffle(routes)
            for key,w,k in routes:
                trial[key] = time_ms(lambda: subprocess.run([args.cpu,"--pack",src,w,k],check=True))
            start = time.perf_counter_ns()
            ms = subprocess.check_output([args.packed_simt,windows,rates,"0"],text=True)
            trial["packed_gpu_process_ms"] = (time.perf_counter_ns()-start)/1e6
            trial["packed_kernel_event_ms"] = float(ms)
            trial["python_csv_merge_ms"] = time_ms(lambda: materialize(keys,rates,result))
            trial["packed_total_ms"] = sum(trial[k] for k in ("pack_to_files_ms","packed_gpu_process_ms","python_csv_merge_ms"))
            if file_info(result)[0] != expected: raise ValueError("packed result mismatch")
            cold = Worker(args.exe,"simt",False)
            trial["stream_cold_ms"],_,batches,segments,trial["stream_init_ms"] = cold.run(src,result)
            if file_info(result)[0] != expected: raise ValueError("cold result mismatch")
            samples.append(trial)
        warm = Worker(args.exe,"simt",True)
        try:
            warm.run(src,result)
            for trial in samples:
                trial["stream_warm_ms"] = warm.run(src,result)[0]
                if file_info(result)[0] != expected: raise ValueError("warm result mismatch")
        finally:
            warm.close()
        report.append(dict(input=str(src),rows=rows,verified_sha256=expected,
             old_intermediate_bytes=sum(x.stat().st_size for x in (windows,keys,rates)),
             old_host_to_device_bytes=windows.stat().st_size, new_intermediate_bytes=0,
             new_host_to_device_bytes=rows*36+(segments+batches)*4,
             median_ms={k:statistics.median(t[k] for t in samples) for k in samples[0]},samples=samples))
    target = run/"profile.json"
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
