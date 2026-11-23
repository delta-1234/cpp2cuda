"""Generate a CUDA replay in an empty directory using the configured model."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from api_agent import OfflineRun, ROOT
from model_api import complete, endpoint, settings
from probe import inventory

FILES = ("gpu.hpp", "kernels.cu", "host.cpp")
SYSTEM = """Write a complete GPU implementation of the given CPU factor replay. You have no tools or internet.
The two input files are the CPU translation unit. The original factor_replay.hpp will also be available to host.cpp;
you may reuse its parsing, validation and ordered replay callbacks. Never compute the factor on the host or invoke
the CPU executable. Implement two distinct and meaningful native CUDA kernels: SIMT and a tiled parallel reduction
using shared memory and synchronization. Actual GPU results must supply all numerical CSV columns. Preserve
integer width, warmup, eviction, multi-symbol input order, validation, formatting and NA. Process large inputs
in bounded batches, reuse device allocations, avoid intermediate files. Both modes must handle partial batches.
Return exactly three complete source files as blocks in this format (no commentary):
<<<gpu.hpp>>>
file contents
<<<END>>>
<<<kernels.cu>>>
file contents
<<<END>>>
<<<host.cpp>>>
file contents
<<<END>>>
In a repair turn return only complete changed files in the same format. No build scripts or extra files.
The executable CLI is: replay simt|tile INPUT.csv OUTPUT.csv. Unknown modes, bad inputs and CUDA failures
must give a nonzero exit. Input generation/packing are reference utilities, not required CLI options.
Use C++17 for host.cpp and nvcc C++14 for kernels.cu; gpu.hpp must interoperate. Host compiler gets CUDA
include and runtime paths. Use only the supplied header, toolkit and standard library."""
BLOCK = re.compile(r"^<<<(gpu\.hpp|kernels\.cu|host\.cpp)>>>\r?\n(.*?)^<<<END>>>[ \t]*\r?$", re.M | re.S)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Generation(OfflineRun):
    def __init__(self, args, cfg):
        super().__init__(ROOT, cfg, args, "from-cpp-")
        self.audit = {"status": "running", "model": cfg["model"], "attempts": []}

    def execute(self):
        from verify_generated import Evaluator
        source = self.args.source.resolve(strict=True)
        header = source.with_name("factor_replay.hpp")
        if source.name != "OrderFillRateSpread.cpp":
            raise ValueError("this evaluator expects OrderFillRateSpread.cpp")
        inputs = {p.name: p.read_text() for p in (source, header)}
        self.audit["input_sha256"] = {p.name: digest(p) for p in (source, header)}
        machine = inventory()
        self.audit["environment"] = machine
        self.record()
        devices = machine["driver"]["devices"]
        if not devices or not machine["nvcc"] or not machine["runtime_headers"]:
            raise RuntimeError("CUDA device, nvcc or runtime headers unavailable; see environment")
        evaluator = Evaluator(self, source, machine)
        evaluator.prepare()
        messages = [{"role": "user", "content": json.dumps({
            "hardware": devices[0], "toolkit": machine["nvcc_version"]["output"], "sources": inputs})}]
        system = SYSTEM
        if self.args.per_file:
            system = system.replace("Return exactly three complete source files as blocks in this format (no commentary):",
                                    "Return ONLY the single complete file requested each turn, using its block format:")
            messages[0]["content"] += "\nStart with gpu.hpp only. Design a compact interface. Keep your answer concise."
        for step in range(1, self.args.attempts+1):
            print(f"Model generation attempt {step}", flush=True)
            answer = self.scrub(complete(self.cfg, system, messages, timeout=self.args.model_timeout, stream=self.args.per_file))
            response = self.dir/f"model-{step}.txt"
            response.write_text(answer)
            entry = {"attempt": step, "response_sha256": digest(response)}
            self.audit["attempts"].append(entry)
            try:
                blocks = list(BLOCK.finditer(answer))
                names = [m[1] for m in blocks]
                if not names or len(names) != len(set(names)):
                    raise ValueError("use one complete block per file with the specified delimiters")
                for m in blocks:
                    if not m[2].strip():
                        raise ValueError("empty source file")
                    (self.work/m[1]).write_text(m[2])
                missing = [p for p in FILES if not (self.work/p).is_file()]
                if missing:
                    next_files = missing[:1] if self.args.per_file else missing
                    entry["received"] = names
                    self.record()
                    messages.extend([{"role": "assistant", "content": answer},
                                     {"role": "user", "content": "Now write complete source for: "+", ".join(next_files)}])
                    continue
                entry["file_sha256"] = {p: digest(self.work/p) for p in FILES}
                evaluator.build()
                entry["validation"] = evaluator.check()
            except (ValueError, RuntimeError) as exc:
                error = self.scrub(str(exc))
                entry["error"] = error
                self.record()
                print(error[:700], flush=True)
                messages.extend([{"role": "assistant", "content": answer},
                                 {"role": "user", "content": "Repair the candidate. Diagnostic:\n"+error[-5000:]}])
                continue
            self.audit["benchmark"] = evaluator.benchmark(dict.fromkeys([32768, 131072, *self.args.extra_orders]))
            self.audit.update(status="verified", source_sha256=entry["file_sha256"])
            return
        raise RuntimeError("generation attempt limit reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--per-file", action=argparse.BooleanOptionalAction, default=True,
                        help="request one full source file per streaming API call (default)")
    parser.add_argument("--attempts", type=int, default=7)
    parser.add_argument("--extra-orders", type=int, nargs="*", default=[], help="extra simulated order counts for timing")
    parser.add_argument("--model-timeout", type=int, default=600)
    parser.add_argument("--stage-timeout", type=int, default=300)
    args = parser.parse_args()
    if min(args.attempts, args.stage_timeout, args.model_timeout) < 1:
        parser.error("attempts and timeouts must be positive")
    if any(n < 1 or n > 4194304 for n in args.extra_orders):
        parser.error("extra order counts must be between 1 and 4194304")
    cfg = settings()
    endpoint(cfg)
    run = Generation(args, cfg)
    print(run.dir, flush=True)
    try:
        run.execute()
        print("Verified", run.work, flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        run.audit.update(status="failed", error=run.scrub(str(exc)))
        print(run.audit["error"], file=sys.stderr)
        return 1
    finally:
        run.record()
    return 0


if __name__ == "__main__":
    sys.exit(main())
