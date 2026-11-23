"""Independent oracle, GPU launch checks and end-to-end timings for generated ports."""
import json
import os
import re
import shutil
import statistics
import sys
from pathlib import Path

from check import compare, edge_input
from probe import cupti_paths
from scale import file_info

MODES = ("simt", "tile")
LARGE = "orders-131072"
TIMED = """import json, subprocess, sys, time
start = time.perf_counter_ns()
with open(sys.argv[1], 'wb') as out:
    subprocess.run(sys.argv[2:], stdout=out, check=True)
print(json.dumps((time.perf_counter_ns()-start)/1e6))"""


class Evaluator:
    def __init__(self, run, source, env):
        self.run, self.work, self.source = run, run.work, source
        self.oracle = run.dir/"oracle"
        self.toolkit = Path(env["nvcc"]).resolve().parents[1] if env.get("nvcc") else None
        devices = env.get("driver", {}).get("devices", [])
        self.arch = devices[0]["cc"] if devices else None
        self.cxx = env["host_cxx"]
        self.sanitizer = env["compute-sanitizer"]
        self.offline = run.offline
        self.gold = {}
        self.last_case = None

    def prepare(self, with_trace=True):
        shutil.copyfile(self.source.with_name("factor_replay.hpp"), self.work/"factor_replay.hpp")
        (self.oracle/"runs").mkdir(parents=True,exist_ok=True)
        for path in (self.source, self.source.with_name("factor_replay.hpp")):
            shutil.copyfile(path, self.oracle/path.name)
        self.offline("oracle-build", [self.cxx, "-std=c++17", "-O3", self.source.name, "-o", "runs/cpu"], self.oracle)
        shutil.copyfile(self.run.root/"sample_ticks.csv", self.work/"sample.csv")
        edge_input(self.work/"edge.csv")
        with (self.work/"edge.csv").open() as src:
            header = src.readline()
            lines = src.readlines()
        (self.work/"header-only.csv").write_text(header)
        (self.work/"warmup499.csv").write_text(header + "".join(lines[:599]))
        # 99 orders and 500 fills: only the order-window gate is closed.
        (self.work/"short-window.csv").write_text(header+"".join(lines[:99]+lines[100:600]))
        (self.work/"overfill.csv").write_text(header+"1,X,B,100,1,1,0\n1,X,S,100,2,0,2\n1,X,T,100,2,1,2\n")
        (self.work/"bad-time.csv").write_text(header+"2,X,B,100,2,1,0\n1,X,S,100,2,0,2\n")
        (self.work/"unknown.csv").write_text(header+"1,X,T,100,1,3,4\n")
        for n, seed in [(32768, 7319), (131072, 15013)]:
            self.simulate(n, seed)
        for case in ("header-only", "warmup499", "short-window", "edge", "sample", "orders-32768", LARGE):
            self.reference(case)
        if file_info(self.gold["sample"]) != file_info(self.run.root/"factor_output.csv"):
            raise RuntimeError("CPU sample differs from frozen reference")
        if with_trace:
            self.prepare_trace()

    def prepare_trace(self):
        if not self.sanitizer:
            raise RuntimeError("Compute Sanitizer is required for GPU verification")
        cupti = cupti_paths(self.toolkit)
        if not all(cupti.values()):
            raise RuntimeError("CUPTI development headers and libcupti.so are required for launch tracing")
        shutil.copyfile(self.run.root/"workflow/trace_launch.cpp", self.oracle/"trace_launch.cpp")
        self.offline("trace-build", [self.cxx, "-std=c++14", "-shared", "-fPIC", "trace_launch.cpp",
                     "-I"+cupti["include"], "-I"+str(self.toolkit/"include"), "-L"+cupti["library"],
                     "-Wl,-rpath,"+cupti["library"], "-lcupti", "-o", "runs/trace.so"], self.oracle)
        shutil.copyfile(self.oracle/"runs/trace.so", self.work/"trace.so")

    def simulate(self, orders, seed):
        filename = f"orders-{orders}.csv"
        self.offline("generate-"+str(orders), ["bash", "-ec", 'runs/cpu --generate-scale "$1" "$2" > "runs/$3"',
                     "_", str(seed), str(orders), filename], self.oracle)
        (self.oracle/"runs"/filename).rename(self.work/filename)

    def reference(self, case):
        filename = case+".csv"
        # Independent mount roots share only read-only fixture bytes.
        (self.oracle/filename).unlink(missing_ok=True)
        os.link(self.work/filename, self.oracle/filename)
        self.offline("reference-"+case, ["bash", "-ec", 'runs/cpu "$1" > "runs/$1"', "_", filename], self.oracle)
        self.gold[case] = self.oracle/"runs"/filename

    def build(self):
        shutil.copyfile(self.run.root/"workflow/build_generated.sh", self.work/"build_generated.sh")
        self.offline("build-generated", ["bash", "build_generated.sh", ".", "runs", ".", self.arch])

    def output(self, name):
        path = self.work/"runs"/name
        if path.resolve() != path or not path.is_file():
            raise ValueError("missing or invalid generated output "+name)
        return path

    def validate(self, case, output):
        actual = self.output(output)
        expected = self.gold[case]
        sha, rows = file_info(actual)
        if (sha, rows) != file_info(expected):
            compare(expected, actual)
            raise ValueError(f"{case}: numeric tolerance passed but output differs in byte formatting or rounding")
        return {"rows": rows, "sha256": sha}

    def test_case(self, case, mode):
        self.last_case = {"case":case,"mode":mode}
        self.offline("focused-"+mode+"-"+case,["runs/replay",mode,case+".csv","runs/focused.csv"])
        return self.validate(case,"focused.csv")

    def expect_rejection(self, name, cmd):
        self.offline(name, ["bash", "-c",
                     '"$@"; rc=$?; test "$rc" -gt 0 && test "$rc" -lt 126', "_", *cmd])

    def check(self):
        cases, launches, rejected = {}, {}, {}
        self.expect_rejection("reject-unknown-mode", ["runs/replay", "unrecognized", "edge.csv", "runs/invalid.csv"])
        for mode in MODES:
            self.expect_rejection("require-device-"+mode, ["env", "CUDA_VISIBLE_DEVICES=", "runs/replay",
                                  mode, "edge.csv", "runs/invalid.csv"])
            for case in self.gold:
                self.last_case = {"case":case,"mode":mode}
                output = f"{mode}-{case}.csv"
                cmd = ["runs/replay", mode, case+".csv", "runs/"+output]
                if case == "sample":
                    log = self.offline("trace-"+mode, ["env", "LD_PRELOAD=/work/trace.so", *cmd])
                    symbols = re.findall(r"^CUPTI_LAUNCH 0 (\S+)$", log, re.M)
                    if not symbols or "(unknown)" in symbols:
                        raise ValueError("no successful named CUDA kernel launch recorded for "+mode)
                    launches[mode] = symbols
                else:
                    self.offline(mode+"-"+case, cmd)
                cases.setdefault(case, {})[mode] = self.validate(case, output)
            self.last_case = None
            rejected[mode] = []
            for case in ("overfill", "unknown", "bad-time"):
                self.expect_rejection("reject-"+mode+"-"+case, ["runs/replay", mode, case+".csv", "runs/invalid.csv"])
                rejected[mode].append(case)
            self.offline("memcheck-"+mode, [self.sanitizer, "--tool", "memcheck", "--error-exitcode", "9",
                         "runs/replay", mode, "edge.csv", "runs/memcheck.csv"])
            self.validate("edge", "memcheck.csv")
        if set(launches["simt"]) == set(launches["tile"]):
            raise ValueError("both modes launched the same kernel symbols; implement distinct kernels")
        return {"byte_exact": cases, "successful_gpu_launches": launches, "invalid_inputs_rejected": rejected,
                "compute_sanitizer_memcheck": list(MODES), "no_device_rejected": list(MODES), "unknown_mode_rejected": True}

    def benchmark(self, orders=(32768, 131072)):
        report = {}
        for n in orders:
            case = f"orders-{n}"
            if case not in self.gold:
                self.simulate(n, 7319)
                self.reference(case)
            raw = {mode: [] for mode in ("cpu", *MODES)}
            for offset in range(3):
                for i in range(3):
                    mode = ("cpu", *MODES)[(i+offset)%3]
                    filename = case+".csv"
                    if mode == "cpu":
                        work, target, cmd = self.oracle, "runs/timed.csv", ["runs/cpu", filename]
                    else:
                        work, target = self.work, "/dev/null"
                        cmd = ["runs/replay", mode, filename, "runs/timed.csv"]
                    log = self.offline("timed-"+mode, [sys.executable, "-c", TIMED, target, *cmd], work)
                    raw[mode].append(round(json.loads(log.splitlines()[-1]), 3))
                    if mode == "cpu":
                        if file_info(self.oracle/"runs/timed.csv") != file_info(self.gold[case]):
                            raise ValueError("timed CPU output mismatch")
                    else:
                        self.validate(case, "timed.csv")
            report[case] = {"events": file_info(self.work/(case+".csv"))[1], "forecasts": file_info(self.gold[case])[1],
                            "samples_ms": raw, "median_ms": {m: statistics.median(v) for m, v in raw.items()}}
        return report
