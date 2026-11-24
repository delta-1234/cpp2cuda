"""Run inside the private tool namespace; extract GPU timestamps and launch resources from Nsight."""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


def entry_hashes(cuobjdump, executable, launches):
    """Match launch-trace symbols to SASS in this exact binary. Compare bytes, not linker names."""
    names = sorted({symbol for mode in launches.values() for symbol in mode})
    result = subprocess.run([cuobjdump, "--dump-sass", "--function", ",".join(names), executable],
                            capture_output=True, text=True, check=True)
    words = {}
    current = None
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*Function : (\S+)\s*",line)
        if match:
            current = match[1]
            words.setdefault(current, [])
        elif current in words:
            words[current].extend(re.findall(r"/\* (0x[0-9a-fA-F]+) \*/",line))
    if any(not words.get(name) for name in names):
        raise RuntimeError("cuobjdump returned no instruction bytes for a traced CUDA entry")
    hashes = {name:hashlib.sha256(" ".join(words[name]).encode()).hexdigest() for name in names}
    return {mode:sorted(hashes[name] for name in set(symbols)) for mode,symbols in launches.items()}


def measure(nsys, mode, args):
    output = "runs/actual"
    named = "{output}" in args
    Path(output).unlink(missing_ok=True)
    argv = [output if a == "{output}" else a for a in args]
    # Nsight has its own stdout. Preserve the candidate's exact file/stdout contract.
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp)/"capture")
        subprocess.run([nsys, "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda",
                        "--export=sqlite", "-o", prefix, "sh", "-c",
                        'out=$1; shift; exec "$@" > "$out"', "_", os.devnull if named else output,
                        "runs/replay", mode, *argv], check=True)
        with sqlite3.connect("file:"+prefix+".sqlite?mode=ro", uri=True) as db:
            rows = db.execute("""
                SELECT s.value, COUNT(*), SUM(k.end-k.start), k.blockX,k.blockY,k.blockZ,
                       k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.gridX,k.gridY,k.gridZ
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                GROUP BY k.demangledName,k.blockX,k.blockY,k.blockZ,k.registersPerThread,
                         k.staticSharedMemory,k.dynamicSharedMemory,k.gridX,k.gridY,k.gridZ ORDER BY SUM(k.end-k.start) DESC
                """).fetchall()
    if not rows or sum(r[2] for r in rows) <= 0:
        raise ValueError("no timed CUDA kernels for this case")
    launches = sorted((r[0],r[1],*r[3:]) for r in rows)
    return {"launch_config_sha256":hashlib.sha256(json.dumps(launches).encode()).hexdigest(),
            "total_kernel_us": sum(r[2] for r in rows)/1000,
            "kernels":[{"name":r[0],"launches":r[1],"us":r[2]/1000,"block":r[3:6],
                        "registers_per_thread":r[6],"static_shared_bytes":r[7],
                        "dynamic_shared_bytes":r[8],"grid":r[9:12]} for r in rows[:20]]}


if __name__ == "__main__":
    if sys.argv[1] == "--fingerprint":
        print(json.dumps(entry_hashes(sys.argv[2], "runs/replay", json.loads(sys.argv[3]))))
    else:
        print(json.dumps(measure(sys.argv[1], sys.argv[2], sys.argv[3:])))
