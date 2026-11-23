"""CUDA tile DSL implementation for systems with a local Triton + PyTorch stack."""
import argparse

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def rate_spread(windows, out, rows, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 128)
    offsets = row[:, None] * 256 + col[None, :]
    valid = row[:, None] < rows
    v = tl.load(windows + offsets, mask=valid, other=0).to(tl.int64)
    f = tl.load(windows + offsets + 128, mask=valid, other=0).to(tl.int64)
    av = tl.sum(tl.where(v > 0, v, 0), axis=1)
    af = tl.sum(tl.where(v > 0, f, 0), axis=1)
    bv = tl.sum(tl.where(v < 0, -v, 0), axis=1)
    bf = tl.sum(tl.where(v < 0, f, 0), axis=1)
    ask = af.to(tl.float64) / tl.maximum(av, 1).to(tl.float64)
    bid = bf.to(tl.float64) / tl.maximum(bv, 1).to(tl.float64)
    tl.store(out + row * 3, tl.where(av > 0, ask, -2.), mask=row < rows)
    tl.store(out + row * 3 + 1, tl.where(bv > 0, bid, -2.), mask=row < rows)
    tl.store(out + row * 3 + 2, tl.where((av > 0) & (bv > 0), ask - bid, -2.), mask=row < rows)


def run(source, target, device):
    torch.cuda.set_device(device)
    elapsed = 0.
    warmed = False
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with open(source, "rb") as src, open(target, "wb") as dst:
        while payload := src.read(4096 * 1024):
            batch = np.frombuffer(payload, dtype="<i4").reshape(-1, 256).copy()
            rows = len(batch)
            data = torch.from_numpy(batch).to(device=f"cuda:{device}")
            out = torch.empty((rows, 3), device=data.device, dtype=torch.float64)
            grid = (triton.cdiv(rows, 8),)
            if not warmed:
                rate_spread[grid](data, out, rows, BLOCK=8)
                torch.cuda.synchronize()
                warmed = True
            start.record()
            rate_spread[grid](data, out, rows, BLOCK=8)
            stop.record()
            stop.synchronize()
            elapsed += start.elapsed_time(stop)
            out.cpu().numpy().tofile(dst)
    print(elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("device", type=int)
    args = parser.parse_args()
    run(args.source, args.target, args.device)
