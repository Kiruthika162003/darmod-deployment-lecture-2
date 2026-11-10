"""Why batching wins, reduced to the one operation it comes down to.

A language model writes one token at a time. At each step every waiting
request needs the same matrix multiply against the same weights. Serving
them one after another does that multiply once per request. Batching
stacks the requests into a single taller matrix and does it once.

The arithmetic is identical either way. The difference is that one big
multiply reads the weights from memory once, keeps the processor fed, and
hands the hardware a shape it was designed for, while a loop of small
multiplies re-reads the same weights on every pass and spends most of its
time waiting on memory.

This is the mechanism behind continuous batching in vLLM. It runs on CPU
with numpy so it works anywhere, which means the ratio you get is a
demonstration of the principle rather than a GPU benchmark. On a real GPU
serving a real model the gap is larger, not smaller.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def unbatched(weights: np.ndarray, requests: np.ndarray) -> float:
    """One multiply per request, the way naive serving does it."""
    started = time.perf_counter()
    for row in requests:
        row @ weights
    return time.perf_counter() - started


def batched(weights: np.ndarray, requests: np.ndarray) -> float:
    """One multiply for every request at once."""
    started = time.perf_counter()
    requests @ weights
    return time.perf_counter() - started


def benchmark(batch: int, hidden: int, vocab: int, repeats: int) -> None:
    rng = np.random.default_rng(0)
    weights = rng.standard_normal((hidden, vocab), dtype=np.float32)
    requests = rng.standard_normal((batch, hidden), dtype=np.float32)

    # One pass of each first, so neither result includes first-touch page
    # faults or a cold cache.
    unbatched(weights, requests)
    batched(weights, requests)

    loop_seconds = sum(unbatched(weights, requests) for _ in range(repeats))
    batch_seconds = sum(batched(weights, requests) for _ in range(repeats))

    operations = batch * repeats
    loop_rate = operations / loop_seconds
    batch_rate = operations / batch_seconds

    print(f"\nOne decode step, {batch} requests waiting, {repeats} repeats")
    print(f"Weights {hidden} x {vocab} float32, {weights.nbytes / 1e6:.1f} MB\n")
    print(f"{'approach':<28}{'ops/sec':>14}{'per op':>12}")
    print("-" * 56)
    print(f"{'one at a time':<28}{loop_rate:>14,.0f}{loop_seconds / operations * 1e6:>10.1f}us")
    print(f"{'batched together':<28}{batch_rate:>14,.0f}{batch_seconds / operations * 1e6:>10.1f}us")
    print("-" * 56)
    print(f"Same total arithmetic. Batched is {batch_rate / loop_rate:.1f}x faster.")
    print("Measured on this machine just now; a GPU widens the gap.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=64, help="requests in flight")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)

    benchmark(args.batch, args.hidden, args.vocab, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
