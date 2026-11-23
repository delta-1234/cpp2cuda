"""NVIDIA cuTile backend. Both inputs to a tile are dense [8, 128] int32."""
import argparse
import sys

import cupy as cp
import cuda.tile as ct
import numpy as np


@ct.kernel
def rate_spread(volumes, fills, ask_out, bid_out, spread_out):
    block = ct.bid(0)
    v = ct.load(volumes, index=(block, 0), shape=(8, 128))
    f = ct.load(fills, index=(block, 0), shape=(8, 128))
    # Widen before reducing: 100 valid int32 quantities can overflow int32.
    v = ct.astype(v, ct.int64)
    f = ct.astype(f, ct.int64)
    av = ct.sum(ct.where(v > 0, v, 0), axis=1)
    af = ct.sum(ct.where(v > 0, f, 0), axis=1)
    bv = ct.sum(ct.where(v < 0, -v, 0), axis=1)
    bf = ct.sum(ct.where(v < 0, f, 0), axis=1)
    ask = ct.astype(af, ct.float64) / ct.astype(ct.where(av > 0, av, 1), ct.float64)
    bid = ct.astype(bf, ct.float64) / ct.astype(ct.where(bv > 0, bv, 1), ct.float64)
    ct.store(ask_out, index=(block,), tile=ct.where(av > 0, ask, -2.0))
    ct.store(bid_out, index=(block,), tile=ct.where(bv > 0, bid, -2.0))
    ct.store(spread_out, index=(block,), tile=ct.where((av > 0) & (bv > 0), ask - bid, -2.0))


def run(source, target, device):
    cp.cuda.Device(device).use()
    stream = cp.cuda.get_current_stream()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    elapsed = 0.0
    warmed = False
    with open(source, 'rb') as src, open(target, 'wb') as dst:
        while data := src.read(4096 * 2 * 128 * 4):
            windows = np.frombuffer(data, dtype='<i4').reshape(-1, 2, 128)
            rows = len(windows)
            padded = (rows + 7) // 8 * 8
            host = np.zeros((2, padded, 128), dtype=np.int32)
            host[:, :rows, :] = windows.transpose(1, 0, 2)
            vol, filled = cp.asarray(host[0]), cp.asarray(host[1])
            outputs = [cp.empty(padded, dtype=cp.float64) for _ in range(3)]
            args = (vol, filled, *outputs)
            grid = (padded // 8, 1, 1)
            if not warmed:  # Compile and execute once before recording GPU events.
                ct.launch(stream, grid, rate_spread, args)
                stream.synchronize()
                warmed = True
            start.record(stream)
            ct.launch(stream, grid, rate_spread, args)
            stop.record(stream)
            stop.synchronize()
            elapsed += cp.cuda.get_elapsed_time(start, stop)
            np.stack([a[:rows].get() for a in outputs], axis=1).tofile(dst)
    print(elapsed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('device', type=int)
    args = parser.parse_args()
    run(args.source, args.target, args.device)
