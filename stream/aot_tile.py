"""Compile the local Triton tile into an embedded cubin once, outside replay."""
import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource


@triton.jit
def tile_prefix(steps, indices, offsets, out):
    tile = tl.program_id(0)
    start = tl.load(offsets + tile)
    end = tl.load(offsets + tile + 1)
    pos = start + tl.arange(0, 128)
    valid = pos < end
    row = tl.load(indices + pos, mask=valid, other=0)
    addr = steps + 4 * row
    av = tl.cumsum(tl.load(addr, mask=valid, other=0), 0)
    af = tl.cumsum(tl.load(addr+1, mask=valid, other=0), 0)
    bv = tl.cumsum(tl.load(addr+2, mask=valid, other=0), 0)
    bf = tl.cumsum(tl.load(addr+3, mask=valid, other=0), 0)
    ask = af.to(tl.float64) / tl.maximum(av,1).to(tl.float64)
    bid = bf.to(tl.float64) / tl.maximum(bv,1).to(tl.float64)
    tl.store(out+row*3, tl.where(av>0,ask,-2.), mask=valid)
    tl.store(out+row*3+1, tl.where(bv>0,bid,-2.), mask=valid)
    tl.store(out+row*3+2, tl.where((av>0)&(bv>0),ask-bid,-2.), mask=valid)


if __name__ == "__main__":
    signature = dict(steps="*i64", indices="*i32", offsets="*i32", out="*fp64")
    kernel = triton.compile(ASTSource(tile_prefix, signature), options=dict(num_warps=4))
    data = ",".join(map(str,kernel.asm["cubin"]))
    meta = kernel.metadata
    Path(sys.argv[1]).write_text(
        f'alignas(8) static const unsigned char tile_cubin[] = {{{data}}};\n'
        f'constexpr char tile_symbol[] = "{meta.name}";\n'
        f'constexpr int tile_threads = {32*meta.num_warps}, tile_shared = {meta.shared};\n')
