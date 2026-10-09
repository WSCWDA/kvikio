import csv
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name+'.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

replay, prepare = module('replay'), module('prepare_edges')

class FakeBackend:
    def __init__(self):
        self.reads, self.pending, self.peak = [], {}, 0
    def submit(self, page, slot):
        assert slot not in self.pending, 'overwriting an in-flight buffer'
        self.reads.append(page); self.pending[slot] = page
        self.peak = max(self.peak, len(self.pending))
        return (page, slot)
    def complete(self, future, page, slot):
        assert future == (page, slot) and self.pending.pop(slot) == page

class TraceTests(unittest.TestCase):
    def test_lru_and_inflight_coalescing(self):
        b = FakeBackend(); s = replay.Scheduler(2, b)
        stats = s.level([dict(page_id=p) for p in [1,1,2,1,3,2]], 2)
        self.assertEqual(b.reads, [1,2,3,2])
        self.assertEqual(stats['gpu_cache_hits'], 2)
        self.assertLessEqual(b.peak, 2)
        self.assertFalse(b.pending)
        s.level([dict(page_id=2),dict(page_id=3)], 1)
        self.assertEqual(b.reads, [1,2,3,2])

    def test_one_slot_with_large_qd(self):
        b = FakeBackend(); s = replay.Scheduler(1, b)
        s.level([dict(page_id=p) for p in [1,2,1]], 64)
        self.assertEqual(b.reads,[1,2,1]); self.assertEqual(b.peak,1)

    def test_strip_and_pad(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'x.dst', Path(d)/'edges.bin'
            payload = struct.pack('<QQQ', 9, 3, 5)
            src.write_bytes(struct.pack('<QQ',3,0)+payload)
            prepare.prepare(src,dst,4096)
            self.assertEqual(dst.read_bytes(),payload+bytes(4096-len(payload)))
            with self.assertRaises(FileExistsError): prepare.prepare(src,dst,4096)
            src.write_bytes(struct.pack('<QQ',4,0)+payload)
            with self.assertRaises(ValueError): prepare.prepare(src,Path(d)/'bad',4096)

    def test_trace_validation_and_overflow_marker(self):
        with tempfile.TemporaryDirectory() as d:
            pages, levels = Path(d)/'b_run_0_pages.csv', Path(d)/'b_run_0_levels.csv'
            pages.write_text('level,sequence,page_id,page_bytes,edge_accesses\n0,0,2,4096,4\n0,1,2,4096,8\n')
            levels.write_text('level,edge_accesses,unique_pages\n0,12,1\n1,0,0\n')
            with self.assertRaises(ValueError): replay.load_trace(pages,levels)
            (Path(d)/'b_run_0_complete').write_text('complete\n')
            lv, groups, size = replay.load_trace(pages,levels)
            self.assertEqual(size,4096); self.assertEqual(groups[1],[])
            levels.write_text('level,edge_accesses,unique_pages\n0,13,1\n')
            with self.assertRaises(ValueError): replay.load_trace(pages,levels)

if __name__ == '__main__': unittest.main()
