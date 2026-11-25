"""Measure warmed single-thread CPU compute and summed device kernel activity on Linux."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
from statistics import median
import subprocess
import sys
import tempfile

from verify_holdout import generate

EXAMPLE = Path(__file__).resolve().parent
ROOT = EXAMPLE.parents[1]
sys.path.insert(0, str(ROOT / "workflow"))
from probe import driver_devices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(days, width, seed):
    rng = random.Random(seed)
    day = date(2018, 1, 2)
    prices = [rng.randint(600, 40000) for _ in range(width)]
    for _ in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        symbols = list(range(width))
        rng.shuffle(symbols)
        rows = []
        for sym in symbols:
            prev = prices[sym]
            change = rng.randint(-35, 35)
            close = max(1, prev + (prev * change // 1000))
            prices[sym] = close
            rows.append((day.strftime("%Y%m%d"), f"{600000 + sym}.SH", prev,
                         0 if rng.randrange(25) == 0 else close))
        yield rows
        day += timedelta(days=1)


def save_input(path, sessions):
    with path.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["date", "symbol", "prev_close_cent", "close_cent"])
        for rows in sessions:
            writer.writerows(rows)
    return path


def suite(work, seed):
    yield "sessions", EXAMPLE / "data/sessions.csv"
    yield "boundaries", EXAMPLE / "data/boundaries.csv"
    yield "holdout", save_input(work / "holdout.csv", generate(seed))
    shapes = [(days, 256) for days in (1, 2, 4, 8, 16, 32, 64, 256, 1024)]
    for days, width in [*shapes, (64, 1024), (16, 4096)]:
        name = f"{days}d-{width}s"
        yield name, save_input(work / (name + ".csv"), panel(days, width, seed))


def parse_trace(path, repeats):
    # Each launch_all is synchronous and its rank kernel is the last launch.
    # shortName is the Nsight symbol without its C++ template parameters.
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT s.value, k.end - k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName ORDER BY k.start
        """).fetchall()
    samples = {"simt": [], "tile": []}
    groups, names, durations = [], [], []
    for symbol, duration in rows:
        if duration <= 0:
            raise ValueError("nonpositive kernel duration")
        names.append(symbol)
        durations.append(duration)
        if symbol not in ("momentum_rank_simt_kernel", "momentum_rank_tile_kernel"):
            continue
        if (names[0] != "session_flags_kernel" or names.count(names[0]) != 1 or
                names[-3:-1] != ["session_starts_kernel", "momentum_returns_kernel"] or
                not any("Scan" in entry for entry in names[1:-3])):
            raise ValueError(f"incomplete device pipeline: {names}")
        mode = "simt" if symbol == "momentum_rank_simt_kernel" else "tile"
        groups.append(mode)
        samples[mode].append((sum(durations) / 1000, durations[-1] / 1000))
        names, durations = [], []
    order = [mode for i in range(repeats)
             for mode in (("simt", "tile") if i % 2 == 0 else ("tile", "simt"))]
    if names or groups != order:
        raise ValueError(f"incomplete capture: expected {2 * repeats} complete pipelines, got {len(groups)}")
    return samples


def measure(name, path, args, work, nsys):
    with path.open() as src:
        sizes = {}
        for row in csv.DictReader(src):
            day = row["date"]
            sizes[day] = sizes.get(day, 0) + 1
    if not sizes:
        raise ValueError(f"{path} has no rows to compute")
    prefix = work / name
    target = ["taskset", "-c", str(args.core), work / "replay"]
    params = [path, str(args.warmup), str(args.repeat)]
    batch_text, *cpu_times = subprocess.check_output([*target, "cpu", *params], text=True).splitlines()
    cpu = [float(t) for t in cpu_times]
    if len(cpu) != args.repeat or min(cpu) <= 0:
        raise ValueError(f"invalid CPU samples for {name}")
    cmd = [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--export=sqlite",
           "--force-overwrite=true", "-o", prefix, *target, "gpu", *params]
    for attempt in range(1, 4):
        with (work / (name + ".log")).open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        try:
            gpu = parse_trace(work / (name + ".sqlite"), args.repeat)
            break
        except (sqlite3.OperationalError, ValueError):
            if attempt == 3:
                raise
    samples = {"cpu": cpu, **{mode: [us for us, _ in entries] for mode, entries in gpu.items()}}
    medians = {mode: median(values) for mode, values in samples.items()}
    return {"rows": sum(sizes.values()), "sessions": len(sizes), "max_session_rows": max(sizes.values()),
            "input_sha256": digest(path), "all_gpu_outputs_match_cpu": True, "capture_attempts": attempt,
            "cpu_batch_size": int(batch_text), "samples_us": samples, "median_us": medians,
            "speedup_vs_cpu": {m: medians["cpu"] / medians[m] for m in gpu},
            "rank_kernel_only_median_us": {m: median(r for _, r in times) for m, times in gpu.items()}}


def version(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="use a nonempty quotes CSV instead of the built-in suite")
    parser.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=11)
    parser.add_argument("--seed", type=int, default=91028)
    args = parser.parse_args()
    if args.core not in os.sched_getaffinity(0) or min(args.warmup, args.repeat) < 1:
        parser.error("select an available core and positive warmup/repeat counts")
    run_root = ROOT / "runs"
    run_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="momentum-compute-", dir=run_root))
    print(work, flush=True)
    device = driver_devices()
    if not device["devices"]:
        raise RuntimeError("no visible CUDA device")
    sm = device["devices"][0]["cc"]
    toolkit = os.environ.get("CUDA_HOME") or Path(shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc").resolve().parents[1]
    nsys = os.environ.get("NSYS") or "nsys"
    subprocess.run(["bash", ROOT / "workflow/build_port.sh", toolkit, sm, "17", "17",
                    "benchmark", work], cwd=EXAMPLE, check=True)
    inputs = ([(f"input-{i + 1}", p.resolve()) for i, p in enumerate(args.input)]
              if args.input else suite(work, args.seed))
    sources = ["MomentumRank.cpp", "rank_cpu.hpp", "rank_io.hpp", "generated/replay.cu",
               "benchmark/cpu.cpp", "benchmark/compute.cu", "benchmark_compute.py", "verify_holdout.py"]
    report = {"device": device, "nsys": version(nsys, "--version"),
              "nvcc": version(str(Path(toolkit) / "bin/nvcc"), "--version"),
              "host_cxx": version(os.getenv("CXX", "c++"), "--version").splitlines()[0],
              "nvcc_ccbin": os.getenv("NVCC_CCBIN"), "cpu": next(line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
              "core": args.core, "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
              "unit": "us", "cpu_sampling": "Unprofiled separate process; at least 50 ms warmup, calibrate batches to >=500 us, divide each elapsed time by its batch size.",
              "method": "Median warm single-thread CPU compute / median sum of all CUDA kernel activity per factor evaluation. Includes session grouping, CUB scan, returns and rank. Excludes CSV, host staging, transfers, CUDA setup, host launch overhead and gaps between kernels.",
              "sources_sha256": {f: digest(EXAMPLE / f) for f in sources}, "cases": {}}
    for name, path in inputs:
        case = measure(name, path, args, work, nsys)
        report["cases"][name] = case
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, case["median_us"], flush=True)


if __name__ == "__main__":
    main()
