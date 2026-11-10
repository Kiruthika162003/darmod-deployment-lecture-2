"""Three shapes of scaling, measured on whatever machine you run this on.

Vertical means one worker doing everything. Horizontal means several
workers doing it side by side. Async means a queue in front, so work is
accepted immediately and done when a worker frees up.

The first two are about how much machine you have. The third is about
whether anybody is waiting, which is a different question, and combining
async with one of the other two is normal rather than exotic.

One measurement note that matters if you show this to people: starting
operating system processes is slow, especially on Windows, and an earlier
version of this benchmark accidentally measured that startup instead of
the work. The pool is warmed before the clock starts, so what you see is
steady-state throughput of a service already running, which is the state
a production system is actually in.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

# 500 KB hashed repeatedly: enough work that scheduling overhead does not
# dominate, small enough that the whole benchmark finishes in seconds.
PAYLOAD = b"\xa5" * 500_000
ROUNDS = 300


def unit_of_work(_: int) -> float:
    """One job. Returns how long it took, so we can report percentiles."""
    started = time.perf_counter()
    digest = hashlib.sha256()
    for _ in range(ROUNDS):
        digest.update(PAYLOAD)
    digest.hexdigest()
    return time.perf_counter() - started


@dataclass
class Measurement:
    shape: str
    workers: int
    jobs: int
    wall_seconds: float
    latencies: list[float]

    @property
    def throughput(self) -> float:
        return self.jobs / self.wall_seconds

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies) * 1000

    @property
    def p99_ms(self) -> float:
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000


def run_vertical(jobs: int) -> Measurement:
    """One worker, start to finish. The baseline everything else beats."""
    started = time.perf_counter()
    latencies = [unit_of_work(i) for i in range(jobs)]
    return Measurement("vertical", 1, jobs, time.perf_counter() - started, latencies)


def run_horizontal(jobs: int, workers: int) -> Measurement:
    """Several processes side by side, with the pool already warm."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(unit_of_work, range(workers)))  # warm up, discarded
        started = time.perf_counter()
        latencies = list(pool.map(unit_of_work, range(jobs)))
        wall = time.perf_counter() - started
    return Measurement("horizontal", workers, jobs, wall, latencies)


def run_async(jobs: int, workers: int) -> Measurement:
    """A queue with several consumers pulling from it.

    The hashing releases the interpreter lock, so threads genuinely run in
    parallel here. The queue is what makes this async: work is accepted
    the moment it arrives and completed whenever a consumer is free.
    """

    async def main() -> tuple[float, list[float]]:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(jobs):
            queue.put_nowait(i)
        latencies: list[float] = []

        async def consumer() -> None:
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                latencies.append(await asyncio.to_thread(unit_of_work, job))

        await asyncio.to_thread(unit_of_work, -1)  # warm the thread pool
        started = time.perf_counter()
        await asyncio.gather(*(consumer() for _ in range(workers)))
        return time.perf_counter() - started, latencies

    wall, latencies = asyncio.run(main())
    return Measurement("async", workers, jobs, wall, latencies)


SUMMARY = {
    "vertical": "One worker. Answers a single job fastest, gets through the least.",
    "horizontal": "Several workers. Serves the most people at once.",
    "async": "A queue in front. Accepts work fastest, longest tail.",
}


def report(measurements: list[Measurement]) -> None:
    print(f"\n{'shape':<12}{'workers':>9}{'jobs':>7}{'ops/sec':>11}{'p50 ms':>10}{'p99 ms':>10}")
    print("-" * 78)
    for m in measurements:
        print(
            f"{m.shape:<12}{m.workers:>9}{m.jobs:>7}"
            f"{m.throughput:>11.1f}{m.p50_ms:>10.1f}{m.p99_ms:>10.1f}"
        )
        print(f"{'':<12}{SUMMARY[m.shape]}")
    print("-" * 78)
    print("Measured on this machine, this run. Expect different numbers on yours.")


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument(
        "--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1)
    )
    args = parser.parse_args(argv)

    print(f"\nScaling shapes. {args.jobs} jobs, {args.workers} workers where relevant.")
    report(
        [
            run_vertical(args.jobs),
            run_horizontal(args.jobs, args.workers),
            run_async(args.jobs, args.workers),
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
