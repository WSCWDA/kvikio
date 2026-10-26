# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled request-size sensitivity benchmark for G-Route.

Sub-4-KiB logical reads are supported. Their file offsets remain 4-KiB aligned
in the shared engine so that the sweep isolates request-size effects rather
than mixing request size with file-offset alignment.
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
            request_size=size,
            concurrency=args.concurrency,
            reuse=args.reuse,
            hotset_requests=args.hotset_requests,
            label=f"size-{size}",
        )
        for size in args.request_sizes
    ]
    run_context_benchmark(
        benchmark_name="size-sensitivity",
        args=args,
        contexts=contexts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "G-Route request-size sensitivity benchmark. Request size varies "
            "while logical concurrency and reuse are fixed. Sub-4-KiB logical "
            "reads use page-aligned file offsets."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--request-sizes",
        nargs="+",
        type=parse_bytes,
        default=[
            parse_bytes("512 B"),
            parse_bytes("1 KiB"),
            parse_bytes("2 KiB"),
            parse_bytes("4 KiB"),
            parse_bytes("8 KiB"),
            parse_bytes("16 KiB"),
            parse_bytes("32 KiB"),
            parse_bytes("64 KiB"),
            parse_bytes("128 KiB"),
            parse_bytes("256 KiB"),
            parse_bytes("512 KiB"),
            parse_bytes("1 MiB"),
            parse_bytes("2 MiB"),
            parse_bytes("4 MiB"),
        ],
        help=(
            "Logical request sizes to sweep. Sizes below 4 KiB are supported; "
            "sizes at or above 4 KiB must be 4-KiB aligned."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Fixed number of outstanding logical requests per batch.",
    )
    parser.add_argument(
        "--reuse",
        choices=REUSE_MODES,
        default="low",
        help="Fixed reuse mode for the size sweep.",
    )
    parser.add_argument(
        "--hotset-requests",
        type=int,
        default=32,
        help="Hot-set size when --reuse high; ignored for low reuse.",
    )
    main(parser.parse_args())
