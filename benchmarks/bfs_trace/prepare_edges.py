#!/usr/bin/env python3
"""Strip the 16-byte BaM .dst header and pad payload for aligned file replay."""
import argparse
import os
from pathlib import Path
import struct


def prepare(source, output, page_bytes):
    if page_bytes <= 0 or page_bytes % 4096:
        raise ValueError('page bytes must be a positive multiple of 4096')
    with source.open('rb') as src:
        header = src.read(16)
        if len(header) != 16:
            raise ValueError('missing .dst header')
        count, dtype = struct.unpack('<QQ', header)
        # BaM ignores typeT and uses uint64_t EdgeT; do not infer an enum meaning.
        size = count*8
        if os.fstat(src.fileno()).st_size < size+16:
            raise ValueError('truncated .dst payload')
        with output.open('xb') as dst:
            remaining = size
            while remaining:
                chunk = src.read(min(8*1024*1024, remaining))
                if not chunk:
                    raise IOError('short payload')
                dst.write(chunk); remaining -= len(chunk)
            dst.write(bytes((-size) % page_bytes))
    print(f'{output}: {count} edges, header typeT={dtype}, {size} payload bytes, padded to {page_bytes}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--page-bytes', type=int, default=4096)
    a = p.parse_args()
    prepare(a.source, a.output, a.page_bytes)
