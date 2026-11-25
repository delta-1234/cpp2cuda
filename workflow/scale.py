"""CSV-to-CSV paired cold/warm sweep. Run on the CUDA node after build_stream.sh."""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from example_paths import ORDER_EXAMPLE, SAMPLE_TICKS

ROOT = Path(__file__).resolve().parents[1]
MODES = ("legacy", "cpu", "simt", "tile")
DEFAULT_SIZES = "1400,1500,1600,2000,2600,4096,8192,32768,65536,81920,98304,114688,131072,262144,524288,1048576,2097152"


def file_info(path):
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines - 1


class Worker:
    def __init__(self, exe, mode, warm):
        self.exe, self.mode, self.p = exe, mode, None
        if warm:
            self.p = subprocess.Popen([exe, mode, "--serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1)
            if not self.p.stdout.readline().startswith("READY\t"):
                raise RuntimeError(f"{mode} failed to initialize")

    def run(self, src, dst):
        start = time.perf_counter_ns()
        if self.p:
            self.p.stdin.write(f"{src}\t{dst}\n")
            self.p.stdin.flush()
            reply = self.p.stdout.readline()
        else:
            reply = subprocess.check_output([self.exe, self.mode, src, dst], text=True)
        ms = (time.perf_counter_ns()-start)/1e6
        fields = reply.strip().split("\t")
        if len(fields) != 5 or fields[0] != "OK":
            raise RuntimeError(f"{self.mode} failed: {reply}")
        rows,batches,segments = map(int, fields[1:4])
        return ms, rows, batches, segments, float(fields[4])

    def close(self):
        if self.p:
            self.p.stdin.close()
            if self.p.wait():
                raise RuntimeError(f"{self.mode} worker failed")


def prepare(folder, sizes, seed):
    generator = folder/"generator"
    subprocess.run([os.getenv("CXX","c++"),"-std=c++17","-O3",ROOT/ORDER_EXAMPLE/"OrderFillRateSpread.cpp","-o",generator],check=True)
    # The standalone legacy calculation supplies the oracle for every timed run.
    cases = {}
    for size in sizes:
        path = folder/f"ticks-{size}.csv"
        with path.open("wb") as out:
            subprocess.run([generator, "--generate-scale", str(seed), str(size)], stdout=out,check=True)
        gold = folder/"reference.csv"
        with gold.open("wb") as out:
            subprocess.run([generator, path],stdout=out,check=True)
        input_sha, ticks = file_info(path)
        expected, rows = file_info(gold)
        cases[size] = dict(ticks=ticks, forecasts=rows, input_bytes=path.stat().st_size,
                           input_sha256=input_sha, output_sha256=expected)
        print("prepared", size, ticks, rows,flush=True)
    return cases


def save(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2)+"\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe",type=Path,default=ROOT/"runs/stream/replay")
    p.add_argument("--sizes",default=DEFAULT_SIZES,help="comma-separated submitted order counts")
    p.add_argument("--with-triton",action="store_true",help="include the ahead-of-time Triton tile")
    p.add_argument("--seed",type=int,default=8001)
    p.add_argument("--repeat",type=int,default=5)
    p.add_argument("--phase",choices=("both","cold","warm"),default="both")
    p.add_argument("--core",type=int,help="pin parent and all workers to one logical CPU")
    p.add_argument("--parent",type=Path,default=ROOT/"runs")
    args = p.parse_args()
    sizes = sorted(set(map(int,args.sizes.split(","))))
    if not sizes or min(sizes)<1 or args.repeat<1: p.error("sizes and repeat must be positive")
    if args.core is not None: os.sched_setaffinity(0,{args.core})
    args.parent.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="scale-",dir=str(args.parent.resolve())))
    exe = args.exe.resolve()
    modes = (*MODES,"triton") if args.with_triton else MODES
    cases = prepare(folder,sizes,args.seed)
    report = dict(platform=platform.platform(), seed=args.seed, repeat=args.repeat,
                  affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                  clock="parent perf_counter_ns; fresh process through exit or warm request through reply",
                  directory=str(folder), cases=cases, samples=[])
    output = folder/"scale.json"
    print(output,flush=True)
    save(output,report)
    rng = random.Random(4091)
    # No warm CUDA contexts are left resident during the cold phase.
    for phase in (("cold","warm") if args.phase=="both" else (args.phase,)):
        workers = {}
        try:
            for mode in modes:
                workers[mode] = Worker(exe,mode,phase=="warm")
                if phase=="warm":
                    workers[mode].run(ROOT/SAMPLE_TICKS, folder/f"{mode}.csv")
            for size in sizes:
                timings = {m:[] for m in modes}
                for trial in range(args.repeat):
                    order = list(modes)
                    rng.shuffle(order)
                    for mode in order:
                        dst = folder/f"{mode}.csv"
                        ms,rows,batches,segments,init = workers[mode].run(folder/f"ticks-{size}.csv",dst)
                        if rows != cases[size]["forecasts"] or file_info(dst)[0] != cases[size]["output_sha256"]:
                            raise RuntimeError(f"result mismatch: {phase} {size} {mode} {trial}")
                        timings[mode].append(ms)
                        report["samples"].append(dict(phase=phase,orders=size,trial=trial,mode=mode,
                            ms=ms,batches=batches,segments=segments,init_ms=init))
                save(output,report)
                print(phase,size,{m:round(statistics.median(v),2) for m,v in timings.items()},flush=True)
        finally:
            for w in workers.values(): w.close()
    print("complete",output)


if __name__=="__main__":
    main()
