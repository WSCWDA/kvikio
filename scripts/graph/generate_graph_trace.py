#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate page-reference traces from BaM-format BFS and PageRank workloads.

BaM stores a CSR graph in ``PREFIX.col`` and ``PREFIX.dst``.  Both files have
two uint64 header words.  The emitted trace contains aligned file offsets into
``PREFIX.dst`` and is therefore directly replayable by KvikIO/G-Route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


HEADER_BYTES = 16


@dataclass(frozen=True)
class BamArray:
    path: Path
    header_count: int
    values: np.memmap


def open_bam_array(path: Path) -> BamArray:
    size = path.stat().st_size
    if size < HEADER_BYTES or (size - HEADER_BYTES) % 8:
        raise ValueError(f"invalid BaM uint64 array size: {path} ({size} bytes)")
    header = np.memmap(path, mode="r", dtype="<u8", shape=(2,))
    count = (size - HEADER_BYTES) // 8
    values = np.memmap(
        path, mode="r", dtype="<u8", offset=HEADER_BYTES, shape=(count,)
    )
    header_count = int(header[0])
    # Some public BaM conversion scripts write nnz into both headers, including
    # the .col file.  The payload length is authoritative; CSR consistency is
    # checked later against col[-1] and the .dst payload.
    return BamArray(path, header_count, values)


class TraceWriter:
    def __init__(self, path: Path, max_requests: int | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("wb")
        self.max_requests = max_requests
        self.count = 0
        self.digest = hashlib.sha256()

    @property
    def full(self) -> bool:
        return self.max_requests is not None and self.count >= self.max_requests

    def append(self, offsets: np.ndarray) -> int:
        values = np.asarray(offsets, dtype="<u8")
        if self.max_requests is not None:
            values = values[: max(0, self.max_requests - self.count)]
        payload = values.tobytes(order="C")
        self._file.write(payload)
        self.digest.update(payload)
        self.count += values.size
        return int(values.size)

    def close(self) -> None:
        self._file.close()


def _expand_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Expand half-open integer ranges without a Python loop."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(ends, dtype=np.int64) - starts
    keep = counts > 0
    starts = starts[keep]
    counts = counts[keep]
    if counts.size == 0:
        return np.empty(0, dtype=np.int64)
    total = int(counts.sum())
    prefix = np.cumsum(counts)
    bases = np.repeat(starts - np.concatenate(([0], prefix[:-1])), counts)
    return bases + np.arange(total, dtype=np.int64)


def _page_offsets(
    edge_starts: np.ndarray, edge_ends: np.ndarray, page_size: int
) -> np.ndarray:
    edge_starts = np.asarray(edge_starts, dtype=np.int64)
    edge_ends = np.asarray(edge_ends, dtype=np.int64)
    keep = edge_ends > edge_starts
    first_bytes = HEADER_BYTES + edge_starts[keep] * 8
    last_bytes = HEADER_BYTES + edge_ends[keep] * 8
    first_pages = first_bytes // page_size
    end_pages = (last_bytes + page_size - 1) // page_size
    return _expand_ranges(first_pages, end_pages).astype(np.uint64) * page_size


def generate_bfs(
    col: np.ndarray,
    dst: np.ndarray,
    writer: TraceWriter,
    *,
    source: int,
    page_size: int,
    max_levels: int,
    frontier_chunk: int,
) -> dict:
    vertex_count = len(col) - 1
    if not 0 <= source < vertex_count:
        raise ValueError(f"BFS source {source} is outside [0, {vertex_count})")
    if int(col[-1]) > len(dst):
        raise ValueError("CSR column offsets exceed the destination array")

    visited = np.zeros(vertex_count, dtype=np.bool_)
    visited[source] = True
    frontier = np.asarray([source], dtype=np.int64)
    level_requests: list[int] = []
    visited_count = 1

    for _level in range(max_levels):
        if frontier.size == 0 or writer.full:
            break
        requests_before = writer.count
        next_parts: list[np.ndarray] = []
        for begin in range(0, frontier.size, frontier_chunk):
            vertices = frontier[begin : begin + frontier_chunk]
            starts = np.asarray(col[vertices], dtype=np.int64)
            ends = np.asarray(col[vertices + 1], dtype=np.int64)
            writer.append(_page_offsets(starts, ends, page_size))

            # The traversal itself is offline trace extraction.  Expanding the
            # exact CSR ranges preserves the same frontier dependence as BFS.
            edge_ids = _expand_ranges(starts, ends)
            if edge_ids.size:
                neighbors = np.asarray(dst[edge_ids], dtype=np.int64)
                neighbors = neighbors[neighbors < vertex_count]
                if neighbors.size:
                    unseen = np.unique(neighbors[~visited[neighbors]])
                    if unseen.size:
                        visited[unseen] = True
                        visited_count += int(unseen.size)
                        next_parts.append(unseen)
            if writer.full:
                break
        level_requests.append(writer.count - requests_before)
        frontier = (
            np.unique(np.concatenate(next_parts))
            if next_parts
            else np.empty(0, dtype=np.int64)
        )

    return {
        "levels": len(level_requests),
        "level_requests": level_requests,
        "visited_vertices": visited_count,
        "vertex_count": vertex_count,
    }


def generate_pagerank(
    col: np.ndarray,
    dst: np.ndarray,
    writer: TraceWriter,
    *,
    page_size: int,
    iterations: int,
    frontier_chunk: int,
    alpha: float,
    tolerance: float,
) -> dict:
    edge_count = int(col[-1])
    if edge_count > len(dst):
        raise ValueError("CSR column offsets exceed the destination array")
    vertex_count = len(col) - 1
    degree = np.diff(np.asarray(col, dtype=np.int64))
    value = np.full(vertex_count, 1.0 - alpha, dtype=np.float32)
    delta = np.zeros(vertex_count, dtype=np.float32)
    nonzero = degree > 0
    delta[nonzero] = (1.0 - alpha) * alpha / degree[nonzero]
    residual = np.zeros(vertex_count, dtype=np.float32)
    active = np.arange(vertex_count, dtype=np.int64)
    iteration_requests: list[int] = []
    active_vertices: list[int] = []
    for _ in range(iterations):
        if active.size == 0 or writer.full:
            break
        before = writer.count
        active_vertices.append(int(active.size))
        for begin in range(0, active.size, frontier_chunk):
            vertices = active[begin : begin + frontier_chunk]
            starts = np.asarray(col[vertices], dtype=np.int64)
            ends = np.asarray(col[vertices + 1], dtype=np.int64)
            writer.append(_page_offsets(starts, ends, page_size))

            counts = ends - starts
            edge_ids = _expand_ranges(starts, ends)
            if edge_ids.size:
                neighbors = np.asarray(dst[edge_ids], dtype=np.int64)
                valid = neighbors < vertex_count
                neighbors = neighbors[valid]
                weights = np.repeat(delta[vertices], counts)[valid]
                residual += np.bincount(
                    neighbors, weights=weights, minlength=vertex_count
                ).astype(np.float32)
            if writer.full:
                break
        iteration_requests.append(writer.count - before)
        active = np.flatnonzero(residual > tolerance)
        value[active] += residual[active]
        active_nonzero = active[degree[active] > 0]
        delta.fill(0)
        delta[active_nonzero] = (
            residual[active_nonzero] * alpha / degree[active_nonzero]
        )
        residual.fill(0)
    return {
        "iterations": len(iteration_requests),
        "iteration_requests": iteration_requests,
        "active_vertices": active_vertices,
        "converged": active.size == 0,
        "rank_sum": float(value.sum(dtype=np.float64)),
        "vertex_count": vertex_count,
        "edge_count": edge_count,
    }


def generate(args: argparse.Namespace) -> dict:
    col_file = open_bam_array(Path(str(args.graph) + ".col"))
    dst_file = open_bam_array(Path(str(args.graph) + ".dst"))
    writer = TraceWriter(args.output, args.max_requests)
    try:
        if args.algorithm == "bfs":
            algorithm = generate_bfs(
                col_file.values,
                dst_file.values,
                writer,
                source=args.source,
                page_size=args.page_size,
                max_levels=args.max_levels,
                frontier_chunk=args.frontier_chunk,
            )
        else:
            algorithm = generate_pagerank(
                col_file.values,
                dst_file.values,
                writer,
                page_size=args.page_size,
                iterations=args.iterations,
                frontier_chunk=args.frontier_chunk,
                alpha=args.alpha,
                tolerance=args.tolerance,
            )
    finally:
        writer.close()

    metadata = {
        "format": "groute-graph-trace-v1",
        "algorithm": args.algorithm,
        "graph_prefix": str(args.graph),
        "data_file": str(dst_file.path),
        "page_size": args.page_size,
        "request_count": writer.count,
        "nominal_logical_bytes": writer.count * args.page_size,
        "trace_sha256": writer.digest.hexdigest(),
        "truncated": writer.full,
        **algorithm,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("bfs", "pagerank"), required=True)
    parser.add_argument("--graph", type=Path, required=True, help="BaM graph prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--max-levels", type=int, default=100)
    parser.add_argument("--frontier-chunk", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--max-requests", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("page_size", "max_levels", "frontier_chunk", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.page_size & (args.page_size - 1):
        parser.error("--page-size must be a power of two")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if args.tolerance <= 0.0:
        parser.error("--tolerance must be positive")
    print(json.dumps(generate(args), indent=2))


if __name__ == "__main__":
    main()
