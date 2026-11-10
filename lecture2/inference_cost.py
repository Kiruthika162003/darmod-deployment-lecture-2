"""What each cost knob actually buys, measured rather than asserted.

Three knobs get named whenever LLM serving costs come up: batching,
quantization, and reusing the KV cache. They do not all pay off the same
way, and a demo that shows all three producing a neat speedup on a laptop
would be lying about at least one of them.

So this measures what is genuinely measurable here and says so:

  Batching is a real, large throughput win on any hardware, and you can
  watch it happen below.

  Quantization's honest result on CPU is memory, not speed. Halving the
  width of the weights halves the bytes, and that is measured. The speed
  benefit is a GPU effect: serving is limited by how fast weights stream
  out of memory, so smaller weights mean more tokens per second. Numpy on
  a CPU will not show that, and this script does not pretend it does.

  Reusing a preallocated output buffer avoids allocating a fresh array on
  every step. That is the same idea as keeping the KV cache around instead
  of rebuilding it, and it is small but real.

Memory is the reason quantization matters most in practice. A model that
does not fit on the GPU you can afford costs infinitely more than one that
does, and that is a threshold effect no throughput number captures.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Row:
    label: str
    ops_per_second: float
    weight_megabytes: float
    note: str


def _time(fn, repeats: int) -> float:
    fn()  # warm
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - started


def measure(batch: int, hidden: int, vocab: int, repeats: int) -> list[Row]:
    rng = np.random.default_rng(0)
    weights32 = rng.standard_normal((hidden, vocab), dtype=np.float32)
    weights16 = weights32.astype(np.float16)
    requests = rng.standard_normal((batch, hidden), dtype=np.float32)
    operations = batch * repeats

    def loop() -> None:
        for row in requests:
            row @ weights32

    def batched() -> None:
        requests @ weights32

    def batched_half() -> None:
        requests.astype(np.float16) @ weights16

    output = np.empty((batch, vocab), dtype=np.float32)

    def batched_reuse() -> None:
        np.matmul(requests, weights32, out=output)

    return [
        Row(
            "baseline, one at a time",
            operations / _time(loop, repeats),
            weights32.nbytes / 1e6,
            "one multiply per request",
        ),
        Row(
            "batched",
            operations / _time(batched, repeats),
            weights32.nbytes / 1e6,
            "the knob that pays off on every kind of hardware",
        ),
        Row(
            "batched, half precision",
            operations / _time(batched_half, repeats),
            weights16.nbytes / 1e6,
            "half the memory, and slower here: numpy upcasts float16",
        ),
        Row(
            "batched, reused buffer",
            operations / _time(batched_reuse, repeats),
            weights32.nbytes / 1e6,
            "same idea as keeping the KV cache around",
        ),
    ]


def report(rows: list[Row]) -> None:
    baseline = rows[0].ops_per_second
    print(f"\n{'configuration':<28}{'ops/sec':>13}{'weights':>11}{'vs base':>10}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row.label:<28}{row.ops_per_second:>13,.0f}"
            f"{row.weight_megabytes:>9.1f}MB{row.ops_per_second / baseline:>9.1f}x"
        )
        print(f"{'':<28}{row.note}")
    print("-" * 92)
    print("Independent configurations of the same work, not a cumulative stack.")
    print("Every number measured on this machine just now.")
    print()
    print("Read the half precision row carefully. It halves the memory, which is")
    print("real and is the point of quantization, and it is slower here, because")
    print("numpy has no float16 matmul and upcasts back to float32. On a GPU that")
    print("row is the fastest one. This is why you benchmark on the hardware you")
    print("intend to serve on rather than trusting a table from a slide.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)

    report(measure(args.batch, args.hidden, args.vocab, args.repeats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
