"""Run inside the private tool namespace; extract GPU timestamps and launch resources from Nsight."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


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
                       k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                GROUP BY k.demangledName,k.blockX,k.blockY,k.blockZ,k.registersPerThread,
                         k.staticSharedMemory,k.dynamicSharedMemory ORDER BY SUM(k.end-k.start) DESC
                """).fetchall()
    if not rows or sum(r[2] for r in rows) <= 0:
        raise ValueError("no timed CUDA kernels for this case")
    return {"total_kernel_us": sum(r[2] for r in rows)/1000,
            "kernels":[{"name":r[0],"launches":r[1],"us":r[2]/1000,"block":r[3:6],
                        "registers_per_thread":r[6],"static_shared_bytes":r[7],
                        "dynamic_shared_bytes":r[8]} for r in rows[:20]]}


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1], sys.argv[2], sys.argv[3:])))
