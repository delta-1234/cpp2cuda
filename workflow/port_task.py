"""Source manifests and the deterministic stdio evaluator for general CUDA ports."""
import hashlib
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

from verify_generated import Evaluator, MODES

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.M)
CPP = {".cpp", ".cc", ".cxx"}
HEADERS = {".hpp", ".hh", ".h", ".hxx", ".inl", ".inc", ".cuh"}


def local(root, name):
    path = (root/name).resolve(strict=True)
    if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("path must be a file in the task directory")
    rel = path.relative_to(root)
    if any(part.startswith(".") for part in rel.parts):
        raise ValueError("hidden paths are outside the task manifest")
    return rel.as_posix()


class Task:
    def __init__(self, source, spec, inputs, inspect=False):
        if spec:
            if source or inputs:
                raise ValueError("use a source with optional --input, or a --spec")
            root = spec.resolve().parent
            data = json.loads(spec.read_text())
            entry = local(root, data["source"])
        elif inspect and not source and not inputs:
            root, entry, data = Path.cwd(), None, {}
        else:
            if not source:
                raise ValueError("supply a C++ source or --spec task.json")
            root = source.resolve(strict=True).parent
            entry = local(root, source.name)
            data = {"fixtures":inputs, "cases":[{"name":f"input-{i+1}", "args":[p]} for i,p in enumerate(inputs)]}
        self.root, self.entry = root, entry
        self.units = ([entry] if entry else [])+[local(root,p) for p in data.get("units",[])]
        if any(Path(p).suffix not in CPP for p in self.units):
            raise ValueError("reference translation units must be C++ sources")
        self.files = {}
        pending = self.units+[local(root,p) for p in data.get("headers",[])]
        while pending:
            name = pending.pop()
            if name in self.files:
                continue
            if Path(name).suffix not in CPP | HEADERS:
                raise ValueError("unsupported source/header extension: "+name)
            body = (root/name).read_text()
            self.files[name] = body
            for inc in INCLUDE.findall(body):
                rel = Path(name).parent/inc
                if (root/rel).is_file():
                    pending.append(local(root,rel))
                elif (root/inc).is_file():
                    pending.append(local(root,inc))
        self.fixtures = [local(root,p) for p in data.get("fixtures",[])]
        self.description = data.get("description","")
        self.cases = {}
        self.proposed = {}
        for case in data.get("cases",[]):
            name, args = case["name"], case["args"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+",name) or name in self.cases:
                raise ValueError("case names must be unique labels")
            if not isinstance(args,list) or not all(isinstance(x,str) and "\0" not in x for x in args):
                raise ValueError("case args must be a JSON array of strings")
            self.cases[name] = args
        self.fixed_cases = set(self.cases)
        profile = data.get("profile")
        if profile not in (None, "stdio", "factor"):
            raise ValueError("supported profiles: stdio, factor")
        self.factor = profile=="factor" or (not spec and not inputs and entry=="OrderFillRateSpread.cpp" and "factor_replay.hpp" in self.files)
        if self.factor and ("factor_replay.hpp" not in self.files or self.cases or self.fixtures or len(self.units)!=1):
            raise ValueError("factor profile requires its replay header and built-in reference suite")

    def snapshot(self, dest):
        self.copy(dest,[*self.files,*self.fixtures])
        self.root = dest

    def propose(self, name, args, content):
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",name) or name in self.fixed_cases:
            raise ValueError("case name must be a new label or a previously proposed case")
        if not isinstance(args,list) or not all(isinstance(x,str) and "\0" not in x for x in args):
            raise ValueError("case args must be strings")
        if content is not None and (not isinstance(content,str) or "{input}" not in args):
            raise ValueError("use {input} in args for the supplied text fixture")
        if "{input}" in args and content is None:
            raise ValueError("{input} needs fixture text")
        self.proposed[name] = content
        self.cases[name] = [f"agent-inputs/{name}.txt" if x=="{input}" else x for x in args]

    def copy(self, dest, names):
        for name in names:
            if Path(name).parts[0] in {"port","runs","trace.so","agent-inputs","build_port.sh"}:
                raise ValueError("reserved task path: "+name)
            path = dest/name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in self.files:
                path.write_text(self.files[name])
            else:
                shutil.copyfile(self.root/name,path)


def file_digest(path):
    if path.resolve()!=path or not path.is_file():
        raise ValueError("missing or invalid trial output")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


# Executed inside the trial namespace. The process timer excludes namespace setup and Python startup.
CAPTURE = """import os, subprocess, sys, time
target, *cmd = sys.argv[1:]
named = '{output}' in cmd
if os.path.lexists(target): os.unlink(target)
cmd = [target if x=='{output}' else x for x in cmd]
with open(os.devnull if named else target, 'wb') as out:
    start = time.perf_counter_ns()
    subprocess.run(cmd, stdout=out, check=True)
    print((time.perf_counter_ns()-start)/1e6)
if not os.path.isfile(target): raise SystemExit('expected output file was not created')
"""


class StdioEvaluator(Evaluator):
    def __init__(self, run, task, env):
        super().__init__(run,task.root/task.entry,env)
        self.task = task

    def prepare(self, with_trace=True):
        (self.oracle/"runs").mkdir(parents=True,exist_ok=True)
        self.task.copy(self.oracle,[*self.task.files,*self.task.fixtures])
        self.task.copy(self.work,[*self.task.fixtures, *(p for p in self.task.files if Path(p).suffix in HEADERS)])
        for name, body in self.task.proposed.items():
            if body is not None:
                for base in (self.oracle,self.work):
                    (base/"agent-inputs").mkdir(exist_ok=True)
                    (base/"agent-inputs"/(name+".txt")).write_text(body)
        self.offline("build-reference",[self.cxx,"-O3","-std=c++17","-I.",*self.task.units,"-o","runs/cpu"],self.oracle)
        for name,args in self.task.cases.items():
            target = Path("runs") / ("reference-"+name+".out")
            self.offline("reference-"+name,[sys.executable,"-c",CAPTURE,target,"runs/cpu",*args],self.oracle)
            self.gold[name] = file_digest(self.oracle/target)
        if with_trace:
            self.prepare_trace()

    def compare_output(self, name, work):
        if file_digest(work/"runs/actual") != self.gold[name]:
            expected = self.oracle/"runs"/("reference-"+name+".out")
            actual = work/"runs/actual"
            with expected.open("rb") as ref, actual.open("rb") as got:
                offset = 0
                while True:
                    left, right = ref.read(8192), got.read(8192)
                    if left != right:
                        idx = next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
                        raise ValueError(f"{name}: first difference at byte {offset+idx}; "
                                         f"expected {left[idx:idx+48]!r}, received {right[idx:idx+48]!r}")
                    if not left:
                        raise ValueError(f"{name}: output changed during verification")
                    offset += len(left)

    def test_case(self, case, mode):
        self.last_case = {"case":case,"mode":mode}
        self.offline("focused-"+mode+"-"+case,[sys.executable,"-c",CAPTURE,"runs/actual",
                     "runs/replay",mode,*self.task.cases[case]])
        self.compare_output(case,self.work)
        return {"case":case,"mode":mode,"output_sha256":self.gold[case]}

    def check(self):
        launches = {}
        for mode in MODES:
            observed = set()
            trace_case = None
            for name,args in self.task.cases.items():
                self.last_case = {"case":name,"mode":mode}
                tail = self.offline("verify-"+mode+"-"+name,[sys.executable,"-c",CAPTURE,"runs/actual","env",
                                    "LD_PRELOAD=/work/trace.so","runs/replay",mode,*args])
                self.compare_output(name,self.work)
                names = re.findall(r"^CUPTI_LAUNCH 0 (\S+)$",tail,re.M)
                if names and "(unknown)" not in names:
                    observed.update(names)
                    if trace_case is None:
                        trace_case = name
            self.last_case = None
            if not observed:
                raise ValueError("no successful named CUDA launches for "+mode)
            self.offline("memcheck-"+mode,[sys.executable,"-c",CAPTURE,"runs/actual",self.sanitizer,
                        "--tool","memcheck","--error-exitcode","9","--log-file","runs/memcheck-"+mode+".log",
                        "runs/replay",mode,*self.task.cases[trace_case]])
            self.compare_output(trace_case,self.work)
            launches[mode] = sorted(observed)
        if launches["simt"] == launches["tile"]:
            raise ValueError("the two backends launched identical kernel sets")
        return {"output_sha256":self.gold,"successful_gpu_launches":launches,"memcheck":list(MODES)}

    def benchmark(self):
        result = {}
        for name,args in self.task.cases.items():
            sample = {m:[] for m in ("cpu",*MODES)}
            for i in range(3):
                modes = ["cpu",*MODES]
                for m in modes[i:]+modes[:i]:
                    work = self.oracle if m=="cpu" else self.work
                    cmd = ["runs/cpu"] if m=="cpu" else ["runs/replay",m]
                    log = self.offline("time-"+m,[sys.executable,"-c",CAPTURE,"runs/actual",*cmd,*args],work)
                    self.compare_output(name,work)
                    sample[m].append(round(json.loads(log.splitlines()[-1]),3))
            result[name] = {"samples_ms":sample,"median_ms":{m:statistics.median(s) for m,s in sample.items()}}
        return result
