#!/usr/bin/env python3
"""Replay BaM logical BFS page demands with a shared bounded GPU LRU."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time


def load_trace(pages_path, levels_path):
    marker = Path(str(pages_path).removesuffix('_pages.csv') + '_complete')
    if not marker.is_file() or marker.read_text().strip() != 'complete':
        raise ValueError('missing completion marker; discard incomplete/overflowed traces')
    with open(levels_path, newline='') as f:
        levels = [dict(r) for r in csv.DictReader(f)]
    with open(pages_path, newline='') as f:
        pages = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
    ids = [int(r['level']) for r in levels]
    if ids != list(range(len(ids))) or not ids:
        raise ValueError('levels must be nonempty and consecutive from zero')
    grouped = {i: [] for i in ids}
    size = None
    previous = -1
    for r in pages:
        lv = r['level']
        if lv not in grouped or lv < previous or r['sequence'] != len(grouped[lv]):
            raise ValueError('invalid level/sequence ordering')
        previous = lv
        if r['page_id'] < 0 or not 1 <= r['edge_accesses'] <= 32:
            raise ValueError('invalid page or edge multiplicity')
        if r['page_bytes'] <= 0 or r['page_bytes'] % 4096:
            raise ValueError('page size must be 4KiB aligned')
        size = r['page_bytes'] if size is None else size
        if size != r['page_bytes']:
            raise ValueError('mixed page sizes')
        grouped[lv].append(r)
    if size is None:
        raise ValueError('trace contains no page demands')
    for r in levels:
        rows = grouped[int(r['level'])]
        if sum(x['edge_accesses'] for x in rows) != int(r['edge_accesses']):
            raise ValueError('edge count mismatch (possibly incomplete trace)')
        if len({x['page_id'] for x in rows}) != int(r['unique_pages']):
            raise ValueError('unique page mismatch')
    return levels, grouped, size


class Scheduler:
    """Deterministic batching; miss list is independent of backend timing.

    GPU slots persist across levels. In-flight duplicates share a slot. Flush
    before eviction so a buffer is never reused while a read is in progress.
    """
    def __init__(self, capacity, backend):
        if capacity < 1:
            raise ValueError('GPU cache needs at least one page')
        self.cache = OrderedDict()
        self.free = list(range(capacity))
        self.backend = backend
        self.pending = []
        self.peak = 0

    def drain(self):
        for future, page, slot in self.pending:
            self.backend.complete(future, page, slot)
        self.pending.clear()

    def level(self, rows, qd):
        if qd < 1:
            raise ValueError('qd must be positive')
        misses = hits = 0
        self.peak = 0
        started = time.perf_counter()
        for r in rows:
            page = r['page_id']
            if page in self.cache:
                hits += 1
                self.cache.move_to_end(page)
                continue
            if not self.free:
                self.drain()
                _, slot = self.cache.popitem(last=False)
            else:
                slot = self.free.pop()
            future = self.backend.submit(page, slot)
            self.cache[page] = slot
            self.pending.append((future, page, slot))
            misses += 1
            self.peak = max(self.peak, len(self.pending))
            if len(self.pending) >= qd:
                self.drain()
        self.drain()
        return dict(seconds=time.perf_counter()-started, page_misses=misses,
                    gpu_cache_hits=hits, replay_pending_peak=self.peak)


class GPUBackend:
    def __init__(self, path, mode, page_bytes, slots, workers, gpu, verify):
        import cupy as cp
        import numpy as np
        self.cp, self.np = cp, np
        self.gpu, self.mode, self.size, self.verify = gpu, mode, page_bytes, verify
        self.fd = os.open(path, os.O_RDONLY)  # buffered POSIX; Linux page cache
        self.handle = None
        cp.cuda.Device(gpu).use()
        # Allocate each slot separately for aligned cuFile buffer registration.
        self.buffers = []
        for _ in range(slots):
            allocation = cp.empty(page_bytes+4095, dtype=cp.uint8)
            pad = (-allocation.data.ptr) % 4096
            self.buffers.append(allocation[pad:pad+page_bytes])
        self.pinned = []
        self.host_buffers = []
        self.streams = []
        if mode == 'host-cached':
            for _ in range(slots):
                pinned = cp.cuda.alloc_pinned_memory(page_bytes)
                self.pinned.append(pinned)
                self.host_buffers.append(np.frombuffer(pinned, dtype=np.uint8))
                self.streams.append(cp.cuda.Stream(non_blocking=True))
        cp.cuda.runtime.deviceSynchronize()
        if mode == 'gds':
            import kvikio
            import kvikio.defaults
            kvikio.defaults.set('compat_mode', False)
            self.handle = kvikio.CuFile(path, 'r')
            if not self.handle.is_direct_io_supported():
                raise RuntimeError('file does not support direct I/O')
            for buf in self.buffers:
                kvikio.memory_register(buf)
            self.kvikio = kvikio
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def read(self, page, slot):
        cp = self.cp
        with cp.cuda.Device(self.gpu):
            offset = page*self.size
            buf = self.buffers[slot]
            if self.mode == 'gds':
                # raw_read bypasses KvikIO's request-size threshold.
                n = self.handle.raw_read(buf, size=self.size, file_offset=offset)
                if n != self.size:
                    raise IOError(f'short cuFile read: {n}/{self.size}')
            else:
                host = self.host_buffers[slot]
                n = os.preadv(self.fd, [host], offset)
                if n != self.size:
                    raise IOError(f'short POSIX read: {n}/{self.size}')
                stream = self.streams[slot]
                with stream:
                    cp.cuda.runtime.memcpyAsync(buf.data.ptr, host.ctypes.data,
                        self.size, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                stream.synchronize()

    def submit(self, page, slot):
        return self.pool.submit(self.read, page, slot)

    def complete(self, future, page, slot):
        future.result()
        if self.verify:
            # Correctness-only run: this read warms the host cache and distorts timing.
            actual = self.buffers[slot].get().tobytes()
            expected = os.pread(self.fd, self.size, page*self.size)
            if actual != expected:
                raise AssertionError(f'payload mismatch for page {page}')

    def close(self):
        self.pool.shutdown(wait=True)
        if self.handle is not None:
            for buf in self.buffers:
                self.kvikio.memory_deregister(buf)
            self.handle.close()
        os.close(self.fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pages', required=True, type=Path)
    p.add_argument('--levels', required=True, type=Path)
    p.add_argument('--edge-file', required=True)
    p.add_argument('--mode', choices=['host-cached', 'gds'], required=True)
    p.add_argument('--gpu-cache-pages', type=int, default=1024)
    p.add_argument('--qd', default='trace', help='trace peak per level or fixed integer')
    p.add_argument('--max-qd', type=int, default=64)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--cufile-config', type=Path, help='GDS requires properties.allow_compat_mode=false')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    cufile_config_sha = None
    if a.mode == 'gds':
        if not a.cufile_config:
            p.error('GDS requires --cufile-config with properties.allow_compat_mode=false')
        config_bytes = a.cufile_config.read_bytes()
        config = json.loads(config_bytes)
        if config.get('properties', {}).get('allow_compat_mode') is not False:
            p.error('cuFile compatibility fallback must be disabled for native GDS testing')
        os.environ['CUFILE_ENV_PATH_JSON'] = str(a.cufile_config.resolve())
        cufile_config_sha = hashlib.sha256(config_bytes).hexdigest()
    levels, grouped, size = load_trace(a.pages, a.levels)
    if a.max_qd < 1 or a.gpu_cache_pages < 1 or (a.qd != 'trace' and int(a.qd) < 1):
        p.error('cache pages and QD must be positive')
    max_page = max(r['page_id'] for rows in grouped.values() for r in rows)
    if os.stat(a.edge_file).st_size < (max_page+1)*size:
        p.error('edge file too short; strip header and pad with prepare_edges.py')
    if a.output.exists() or a.output.with_suffix('.json').exists():
        p.error('output already exists; choose next run number')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    backend = GPUBackend(a.edge_file, a.mode, size, a.gpu_cache_pages, a.max_qd, a.gpu, a.verify)
    rows = []
    try:
        scheduler = Scheduler(a.gpu_cache_pages, backend)
        for lv in levels:
            i = int(lv['level'])
            qd = max(1, min(a.max_qd, int(lv['outstanding_page_requests_peak']) if a.qd == 'trace' else int(a.qd)))
            result = scheduler.level(grouped[i], qd)
            result.update(level=i, mode=a.mode, qd=qd,
                          frontier_size=int(lv['frontier_size']), edge_accesses=int(lv['edge_accesses']),
                          physical_bytes=result['page_misses']*size)
            rows.append(result)
    finally:
        backend.close()
    with a.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import cupy
    metadata = dict(mode=a.mode, pages_sha256=hashlib.sha256(a.pages.read_bytes()).hexdigest(),
                    levels_sha256=hashlib.sha256(a.levels.read_bytes()).hexdigest(),
                    gpu_cache_pages=a.gpu_cache_pages, page_bytes=size,
                    qd=a.qd, max_qd=a.max_qd, verified=a.verify,
                    cupy=cupy.__version__, edge_file=os.path.abspath(a.edge_file),
                    edge_file_size=os.stat(a.edge_file).st_size,
                    kernel=os.uname().release, cuda_runtime=cupy.cuda.runtime.runtimeGetVersion(),
                    cuda_driver=cupy.cuda.runtime.driverGetVersion(),
                    cufile_config_sha256=cufile_config_sha,
                    gds_label='cuFile forced, compat fallback disabled; verify direct path with cuFile stats')
    if a.mode == 'gds':
        metadata['kvikio'] = backend.kvikio.__version__
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(a.output)

if __name__ == '__main__':
    main()
