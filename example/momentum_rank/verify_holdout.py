"""Check independently built CPU, SIMT and Tile programs on a seeded holdout."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile


def generate(seed):
    rng = random.Random(seed)
    sessions = []
    for date, count in (("20241104", 2), ("20241105", 127), ("20241106", 128),
                        ("20241107", 129), ("20241108", 1025), ("20241111", 5003)):
        rows = []
        for index in range(count):
            previous = rng.randint(500, 50000)
            close = previous + int(Fraction(previous * rng.randint(-20, 20), 1000))
            if index % 13 == 4:
                previous = 0
            elif index % 17 == 7:
                close = 0
            rows.append((date, f"{600000 + index}.SH", previous, close))
        rng.shuffle(rows)
        sessions.append(rows)
    # Numerical boundary inputs cover an output above INT32_MAX and a
    # negative fractional return close to zero. These are synthetic stress cases.
    sessions.append([("20241112", "TEST-A", 1, 500001),
                     ("20241112", "TEST-B", 999999999, 999999998),
                     ("20241112", "TEST-C", 1000, 998)])
    return sessions


def reference(sessions):
    lines = ["date,symbol,return_bp,rank_ppm"]
    for rows in sessions:
        returns = [int(Fraction((close - prev) * 10000, prev)) if prev and close else None
                   for _, _, prev, close in rows]
        counts = Counter(r for r in returns if r is not None)
        valid = sum(counts.values())
        positions = {}
        first = 1
        for r, ties in sorted(counts.items()):
            midpoint = Fraction(first + first + ties - 1, 2)
            score = (2 * (midpoint - 1) / (valid - 1) - 1) if valid > 1 else 0
            positions[r] = int(score * 1000000)
            first += ties
        for (date, symbol, _, _), r in zip(rows, returns):
            value = "NA,NA" if r is None else f"{r},{positions[r]}"
            lines.append(f"{date},{symbol},{value}")
    return ("\n".join(lines) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True, help="replay executable accepting simt or tile")
    parser.add_argument("--seed", type=int, default=91028)
    args = parser.parse_args()
    sessions = generate(args.seed)
    expected = reference(sessions)
    commands = {"cpu": [args.cpu.resolve()], "simt": [args.gpu.resolve(), "simt"],
                "tile": [args.gpu.resolve(), "tile"]}
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "holdout.csv"
        with path.open("w") as f:
            f.write("date,symbol,prev_close_cent,close_cent\n")
            for session in sessions:
                for date, symbol, prev, close in session:
                    f.write(f"{date},{symbol},{prev},{close}\n")
        for name, cmd in commands.items():
            actual = subprocess.run([*cmd, path], check=True, stdout=subprocess.PIPE).stdout
            if actual != expected:
                raise SystemExit(f"{name}: output differs from independent reference")
        print(json.dumps({"seed": args.seed, "session_sizes": [len(s) for s in sessions],
                          "rows": sum(map(len, sessions)), "matched": list(commands),
                          "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "output_sha256": hashlib.sha256(expected).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
