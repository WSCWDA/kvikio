# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy generic entry point for the controlled G-Route benchmark.

New experiments should prefer the dedicated entry points:

- ``groute_size_bench.py``
- ``groute_concurrency_bench.py``
- ``groute_hotness_bench.py``

This module remains as a compatibility wrapper around
``groute_context_common.py``.
"""

from __future__ import annotations

import argparse

from dask.utils import parse_bytes

from kvikio.benchmarks.groute_context_common import (
    Context,
    REUSE_MODES,
    add_common_arguments,
    run_context_benchmark,
)


def main(args) -> None:
    contexts = [
        Context(
            request_size=args.request_size,
            concurrency=concurrency,
            reuse=reuse,
            hotset_requests=args.hotset_requests,
            label=f"req-{args.request_size}-c-{concurrency}-{reuse}",
        )
        for reuse in args.reuse
        for concurrency in args.concurrency
    ]

    run_context_benchmark(
        benchmark_name="generic-context",
        args=args,
        contexts=contexts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic controlled G-Route context benchmark"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--request-size",
        default=parse_bytes("512 KiB"),
        type=parse_bytes,
        help="Logical request size. Must be 4-KiB aligned.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 16, 64],
        help="Concurrent logical requests per batch.",
    )
    parser.add_argument(
        "--reuse",
        nargs="+",
        choices=REUSE_MODES,
        default=list(REUSE_MODES),
        help="Reuse modes to evaluate.",
    )
    parser.add_argument(
        "--hotset-requests",
        default=32,
        type=int,
        help="Distinct request-sized offsets used by high-reuse mode.",
    )
    main(parser.parse_args())
