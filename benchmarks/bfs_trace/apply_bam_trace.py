#!/usr/bin/env python3
"""Apply BaM BFS instrumentation without changing KvikIO core."""
import argparse
from pathlib import Path
import shutil
import subprocess

BASE = '315fadfc5c5c018a64596157bfac94ecbb7d87a2'

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'BaM source mismatch: expected exactly one occurrence of {old[:80]!r}')
    return text.replace(old, new, 1)

def apply(root):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != BASE:
        raise ValueError(f'expected pinned BaM commit {BASE}, found {head}')
    main = root / 'benchmarks/bfs/main.cu'
    cache = root / 'include/page_cache.h'
    if 'bfs_trace.cuh' in main.read_text():
        raise ValueError('already patched; use a fresh BaM checkout')
    m, c = main.read_text(), cache.read_text()
    m = once(m, '#include <page_cache.h>', '#include "bfs_trace.cuh"\n#include <page_cache.h>')
    start = m.index('void kernel_frontier_coalesce_pc(')
    end = m.index('void kernel_frontier_coalesce_ptr_pc(', start)
    kernel = once(m[start:end], 'const EdgeT next = da->seq_read(i);', 'bfs_trace_page(i);\n                const EdgeT next = da->seq_read(i);')
    m = m[:start] + kernel + m[end:]
    m = once(m, '         // Set root', '         BfsTrace bfs_trace(pc_page_size, (int)type, (int)mem, settings.n_ctrls);\n         // Set root')
    m = once(m, '             level = 0;', '             bfs_trace.run(i);\n             level = 0;')
    m = once(m, '                 uint64_t active = changed_h;', '                 bfs_trace.begin();\n                 uint64_t active = changed_h;')
    m = once(m, '                 iter++;', '                 bfs_trace.finish(level, active);\n                 iter++;')
    m = once(m, "             } while(changed_h);", "             } while(changed_h);\n             bfs_trace.complete();")
    begin = c.index('inline __device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {')
    end = c.index('\n}', begin) + 2
    part = once(c[begin:end], '    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);', '    BFS_TRACE_IO_BEGIN();\n    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);')
    part = once(part, '    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);', '    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);\n    BFS_TRACE_IO_END();')
    c = c[:begin] + part + c[end:]
    c = '#ifndef BFS_TRACE_IO_BEGIN\n#define BFS_TRACE_IO_BEGIN() ((void)0)\n#define BFS_TRACE_IO_END() ((void)0)\n#endif\n' + c
    main.write_text(m); cache.write_text(c)
    shutil.copyfile(Path(__file__).with_name('bfs_trace.cuh'), main.with_name('bfs_trace.cuh'))
    print(f'Patched {root}; reference BaM commit {BASE}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bam', type=Path)
    apply(parser.parse_args().bam.resolve())
