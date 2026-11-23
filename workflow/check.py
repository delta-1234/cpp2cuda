"""Stream the offline reference, the packed windows and the three GPU outputs."""
import csv
import itertools
import math
import struct

HEADER = ["time_ms", "symbol", "trade_count", "ask_fill_rate", "bid_fill_rate", "alpha"]
WINDOW = struct.Struct("<256i")
RATES = struct.Struct("<3d")


def records(stream, layout):
    while data := stream.read(layout.size):
        if len(data) != layout.size:
            raise ValueError("truncated binary record")
        yield layout.unpack(data)


def host_rates(windows):
    for row in records(windows, WINDOW):
        vols, fills = row[:128], row[128:]
        if any(vols[100:]) or any(fills[100:]):
            raise ValueError("nonzero padding")
        av = sum(v for v in vols if v > 0)
        bv = -sum(v for v in vols if v < 0)
        ask = sum(f for v, f in zip(vols, fills) if v > 0) / av if av else -2
        bid = sum(f for v, f in zip(vols, fills) if v < 0) / bv if bv else -2
        yield ask, bid, ask - bid if av and bv else -2


def materialize(keys, source, target, host=False):
    with keys.open() as k, source.open("rb") as src, target.open("w", newline="") as out:
        reader = csv.reader(k)
        if next(reader, []) != HEADER[:3]:
            raise ValueError("bad row-key header")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(HEADER)
        rows = host_rates(src) if host else records(src, RATES)
        for key, values in itertools.zip_longest(reader, rows):
            if key is None or values is None:
                raise ValueError("key/result count mismatch")
            if any(v != -2 and (not math.isfinite(v) or abs(v) > 1) for v in values):
                raise ValueError(f"invalid rate/spread at {key}")
            writer.writerow(key + ["NA" if v == -2 else f"{v:.8f}" for v in values])


def compare(reference, actual):
    with reference.open() as ref, actual.open() as got:
        a, b = csv.reader(ref), csv.reader(got)
        if next(a, []) != HEADER or next(b, []) != HEADER:
            raise ValueError("unexpected result header")
        count = 0
        for count, (x, y) in enumerate(itertools.zip_longest(a, b), 1):
            if x is None or y is None or len(x) != 6 or len(y) != 6 or x[:3] != y[:3]:
                raise ValueError(f"row count/order/key mismatch at output {count}")
            for left, right in zip(x[3:], y[3:]):
                if left == right:
                    continue
                if "NA" in (left, right) or not math.isclose(float(left), float(right), abs_tol=1.000001e-8, rel_tol=0):
                    raise ValueError(f"output {count}, key {x[:3]}: {left} != {right}")
    return count


def edge_input(path):
    # Same timestamp, partial fills, an int32-overflowing window, old references and single-sided windows.
    with path.open("w") as f:
        f.write("time_ms,symbol,type,price_cent,volume,bid_order,ask_order\n")
        def order(i, side):
            bid, ask = (i, 0) if side == 'B' else (0, i)
            f.write(f"34200000,EDGE,{side},1000,2000000000,{bid},{ask}\n")
        def trade(bid, ask):
            f.write(f"34200000,EDGE,T,1000,1000000,{bid},{ask}\n")
        for i in range(1, 101):
            order(i, "S" if i % 2 else "B")
        for _ in range(500):
            trade(2, 1)
        for i in range(101, 201):
            order(i, "S")
        trade(2, 200)
        for i in range(201, 301):
            order(i, "B")
        trade(300, 200)
