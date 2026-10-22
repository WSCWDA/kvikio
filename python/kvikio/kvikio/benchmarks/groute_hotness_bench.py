# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled data-hotness/reuse sensitivity benchmark for G-Route."""

from __future__ import annotations

import argparse

from dask.utils import parse_bytes

from kvikio.benchmarks.groute_context_common import (
    Context,
    add_common_arguments,
    run_context_benchmark,
)


def main(args) -> None:
    contexts = []

    if args.include_low_reuse:
        contexts.append(
            Context(
                request_size=args.request_size,
                concurrency=args.concurrency,
                reuse="low",
                hotset_requests=1,
                label="reuse-low",
            )
        )

    contexts.extend(
        Context(
            request_size=args.request_size,
            concurrency=args.concurrency,
            reuse="high",
            hotset_requests=hotset,
            label=f"hotset-{hotset}",
        )
        for hotset in args.hotset_requests
    )

    run_context_benchmark(
        benchmark_name="hotness-sensitivity",
        args=args,
        contexts=contexts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "G-Route data-hotness sensitivity benchmark. Request size and "
            "logical concurrency are fixed while the request hot set varies."
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
        type=int,
        default=4,
        help="Fixed number of outstanding logical requests per batch.",
    )
    parser.add_argument(
        "--hotset-requests",
        nargs="+",
        type=int,
        default=[4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8],
        help=(
            "Configured hot-set sizes. The CSV reports the actual observed "
            "offset_reuse_ratio, which should be used for analysis/plotting."
        ),
    )
    parser.add_argument(
        "--include-low-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a deterministic zero-reuse baseline (default: enabled).",
    )
    main(parser.parse_args())
