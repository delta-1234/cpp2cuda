#!/usr/bin/env python3
"""Offline CUDA migration gate for OrderFillRateSpread. Requires Python 3.8+."""
import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from example_paths import EXPECTED_OUTPUT, ORDER_EXAMPLE, SAMPLE_TICKS
from check import compare, edge_input, materialize
from probe import inventory

ROOT = Path(__file__).resolve().parents[1]


def execute(args, output=None):
    args = [str(a) for a in args]
    p = subprocess.run(args, stdout=output if output else subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(f"{args[0]} exited {p.returncode}: {p.stderr.decode(errors='replace')[-4000:]}")
    return p.stdout.decode().strip() if output is None else None


def forecast(exe, source, dest):
    with dest.open("wb") as out:
        execute([exe, source], out)


class Case:
    def __init__(self, cpu, source, folder):
        folder.mkdir()
        self.folder, self.source = folder, source
        self.keys, self.windows = folder/"keys.csv", folder/"windows.bin"
        self.ref = folder/"cpu.csv"
        forecast(cpu, source, self.ref)
        self.pack(cpu)
        host = folder/"host_windows.csv"
        materialize(self.keys, self.windows, host, host=True)
        self.rows = compare(self.ref, host)

    def pack(self, cpu):
        execute([cpu, "--pack", self.source, self.windows, self.keys])

    def run_gpu(self, name, cmd, device):
        raw, csv = self.folder/(name+".bin"), self.folder/(name+".csv")
        kernel_ms = float(execute([*cmd, self.windows, raw, device]))
        materialize(self.keys, raw, csv)
        return kernel_ms, csv


def native_build(name, env, device, out):
    args = [env["nvcc"], "-std=c++14", "-O3"]
    if os.environ.get("NVCC_CCBIN"):
        args += ["-ccbin", os.environ["NVCC_CCBIN"]]
    if device:
        cc = int(device["cc"])
        arches = [int(n) for n in re.findall(r"compute_(\d+)\b", env["nvcc_arches"]["output"])]
        if cc in arches or not arches:
            args += [f"-arch=sm_{cc}"]
        else:
            eligible = [v for v in arches if v <= cc]
            if not eligible:
                raise RuntimeError(f"nvcc cannot target this device (sm_{cc})")
            target = max(eligible)
            args += [f"-gencode=arch=compute_{target},code=compute_{target}"]
    if name == "shared_tile":
        args.append("-DSHARED_TILE")
    args += [ROOT/ORDER_EXAMPLE/"cuda/native.cu", "-o", out/name]
    execute(args)
    return [out/name], list(map(str, args))


def workflow(args, out, report):
    env = report["environment"] = inventory()
    print("GPU:", env["driver"], "\nnvcc:", env["nvcc"], flush=True)
    if args.probe_only:
        return 0
    if sys.byteorder != "little":
        raise RuntimeError("packed GPU input requires a little-endian host")
    devices = {d["ordinal"]: d for d in env["driver"]["devices"]}
    device = devices.get(args.device)
    cpu = out/"cpu"
    cc = env["host_cxx"]
    if not cc:
        raise RuntimeError("missing C++17 host compiler")
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    flags = ["-isysroot", sdk] if platform.system() == "Darwin" and sdk.exists() else []
    execute([cc, "-std=c++17", "-O2", *flags, ROOT/ORDER_EXAMPLE/"OrderFillRateSpread.cpp", "-o", cpu])
    main = Case(cpu, args.input, out/"main")
    if args.input.resolve() == (ROOT/SAMPLE_TICKS).resolve():
        compare(ROOT/EXPECTED_OUTPUT, main.ref)
    edge_csv = out/"edge_ticks.csv"
    edge_input(edge_csv)
    edge = Case(cpu, edge_csv, out/"edge")
    if edge.rows != 3:
        raise ValueError("edge case should emit three forecasts")
    # Header-only input also exercises the zero-grid dispatch path.
    empty_csv = out/"empty_ticks.csv"
    empty_csv.write_text("time_ms,symbol,type,price_cent,volume,bid_order,ask_order\n")
    empty = Case(cpu, empty_csv, out/"empty")
    cpu_times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        forecast(cpu, args.input, main.ref)
        cpu_times.append((time.perf_counter()-start)*1000)
    cpu_ms = statistics.median(cpu_times)
    report["cpu"] = {"status": "verified", "rows": main.rows, "edge_rows": edge.rows,
                     "empty_rows": empty.rows, "wall_ms": cpu_times, "median_wall_ms": cpu_ms}
    packed_cpu = out/"packed_cpu"
    execute([cc, "-std=c++17", "-O3", *flags, ROOT/ORDER_EXAMPLE/"packed_cpu.cpp", "-o", packed_cpu])
    packed_raw = main.folder/"packed_cpu.bin"
    packed_times = [float(execute([packed_cpu, main.windows, packed_raw])) for _ in range(args.repeat)]
    packed_csv = main.folder/"packed_cpu.csv"
    materialize(main.keys, packed_raw, packed_csv)
    compare(main.ref, packed_csv)
    packed_ms = statistics.median(packed_times)
    report["cpu"].update(packed_compute_ms=packed_times, median_packed_compute_ms=packed_ms)
    lanes = report["backends"] = {}
    for name in ("simt", "shared_tile", "triton_tile", "cutile"):
        result = lanes[name] = {}
        try:
            if name in ("triton_tile", "cutile"):
                probe = env["tile_import" if name == "cutile" else "triton_import"]
                if probe["code"]:
                    result.update(status="unavailable", reason=probe["output"])
                    continue
                script = "cutile_backend.py" if name == "cutile" else "triton_tile.py"
                cmd = [sys.executable, ROOT/ORDER_EXAMPLE/"cuda"/script]
            else:
                if not env["nvcc"]:
                    result.update(status="unavailable", reason="nvcc not found")
                    continue
                cmd, result["build"] = native_build(name, env, device, out)
            if device is None:
                result.update(status="unavailable", reason=f"no visible CUDA device {args.device}")
                continue
            for case in (edge, empty, main):
                _, actual = case.run_gpu(name, cmd, args.device)
                compare(case.ref, actual)
            wall, kernels = [], []
            for _ in range(args.repeat):
                start = time.perf_counter()
                main.pack(cpu)
                ms, actual = main.run_gpu(name, cmd, args.device)
                wall.append((time.perf_counter()-start)*1000)
                kernels.append(ms)
                compare(main.ref, actual)
            median_wall = statistics.median(wall)
            median_kernel = statistics.median(kernels)
            result.update(status="verified", rows=main.rows, edge_rows=edge.rows, empty_rows=empty.rows,
                          wall_ms=wall, kernel_ms=kernels, median_wall_ms=median_wall,
                          median_kernel_ms=median_kernel, end_to_end_speedup=cpu_ms/median_wall,
                          packed_compute_speedup=packed_ms/median_kernel if median_kernel else None)
        except (OSError, RuntimeError, ValueError) as exc:
            result.update(status="failed", reason=str(exc))
        finally:
            print(name, result["status"], flush=True)
    if any(v["status"] == "failed" for v in lanes.values()):
        return 1
    required = ["simt", "shared_tile"] + (["cutile"] if args.require_cutile else [])
    return 0 if all(lanes[k]["status"] == "verified" for k in required) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/SAMPLE_TICKS)
    p.add_argument("--device", type=int, default=0, help="visible CUDA device ordinal")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--require-cutile", action="store_true", help="also require the NVIDIA cuTile DSL lane")
    p.add_argument("--runs-dir", type=Path, default=ROOT/"runs")
    args = p.parse_args()
    if args.repeat < 1 or args.device < 0:
        p.error("repeat must be positive and device nonnegative")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=str(args.runs_dir.resolve())))
    report = {"host": platform.node(), "input": str(args.input.resolve()), "device_ordinal": args.device,
              "repeat": args.repeat, "require_cutile": args.require_cutile}
    os.environ.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    try:
        status = workflow(args, out, report)
    except (OSError, RuntimeError, ValueError) as exc:
        report["error"] = str(exc)
        print(exc, file=sys.stderr)
        status = 1
    report["exit_code"] = status
    path = out/"report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(path)
    return status


if __name__ == "__main__":
    sys.exit(main())
