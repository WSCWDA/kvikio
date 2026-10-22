# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled logical-concurrency sensitivity benchmark for G-Route."""

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
            reuse=args.reuse,
            hotset_requests=args.hotset_requests,
            label=f"concurrency-{concurrency}",
        )
        for concurrency in args.concurrency
    ]
    run_context_benchmark(
        benchmark_name="concurrency-sensitivity",
        args=args,
        contexts=contexts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "G-Route concurrency sensitivity benchmark. Logical concurrency "
            "varies while request size, reuse, and KvikIO worker count remain fixed."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--request-size",
        type=parse_bytes,
        default=parse_bytes("64 KiB"),
        help="Fixed logical request size.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Outstanding logical requests per batch to sweep.",
    )
    parser.add_argument(
        "--reuse",
        choices=REUSE_MODES,
        default="low",
        help="Fixed reuse mode for the concurrency sweep.",
    )
    parser.add_argument(
        "--hotset-requests",
        type=int,
        default=32,
        help="Hot-set size when --reuse high; ignored for low reuse.",
    )
    main(parser.parse_args())
