#!/usr/bin/env python3
"""Compare paired replay CSVs, rejecting unequal inputs or cache policies."""
import argparse
import csv
import json
from pathlib import Path


def compare(host, gds):
    hm = json.loads(host.with_suffix('.json').read_text())
    gm = json.loads(gds.with_suffix('.json').read_text())
    for key in ['pages_sha256', 'levels_sha256', 'gpu_cache_pages', 'page_bytes',
                'qd', 'max_qd', 'edge_file', 'edge_file_size']:
        if hm[key] != gm[key]:
            raise ValueError(f'unpaired replay: {key} differs')
    if hm['mode'] != 'host-cached' or gm['mode'] != 'gds':
        raise ValueError('expected host-cached followed by gds')
    if hm['verified'] or gm['verified']:
        raise ValueError('verification warms cache; use timing runs without --verify')
    with host.open() as f:
        h = list(csv.DictReader(f))
    with gds.open() as f:
        g = list(csv.DictReader(f))
    if not h or len(h) != len(g):
        raise ValueError('level count mismatch')
    best = 0.0
    for a, b in zip(h, g):
        for key in ['level', 'qd', 'page_misses', 'gpu_cache_hits', 'physical_bytes']:
            if a[key] != b[key]:
                raise ValueError(f'unpaired request/cache plan at level {a["level"]}: {key}')
        ht, gt = float(a['seconds']), float(b['seconds'])
        best += min(ht, gt)
        yield dict(level=a['level'], qd=a['qd'], page_misses=a['page_misses'],
                   host_seconds=ht, gds_seconds=gt, host_over_gds=ht/gt if gt else 0,
                   preferred='host-cached' if ht < gt else 'gds')
    th = sum(float(x['seconds']) for x in h)
    tg = sum(float(x['seconds']) for x in g)
    print(f'Host={th:.6f}s GDS={tg:.6f}s per-level minimum={best:.6f}s')
    print('Per-level minimum is descriptive, not a state-consistent routing oracle.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', type=Path); p.add_argument('gds', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rows = list(compare(a.host, a.gds))
    with a.output.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
