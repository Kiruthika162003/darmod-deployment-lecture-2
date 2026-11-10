"""One service, several customers, and a limit that keeps them apart.

Running a separate copy of a service for every customer is the safest
thing you can do and almost nobody can afford it. So customers share, and
sharing creates a problem that did not exist before: one customer's bad
afternoon becomes everybody's bad afternoon. A load test in production, a
runaway retry loop, a genuinely successful launch. The mechanism does not
matter; the shared capacity gets eaten and everyone else times out.

A token bucket per tenant fixes it. Each customer accumulates tokens at
their own rate up to their own ceiling, and a request without a token is
refused immediately with a 429 rather than being queued behind the flood.
Refusing quickly is the whole point: a rejected request costs almost
nothing, while a queued one holds capacity that a well behaved customer
needed.

The tenants here run at the same time, against the same worker pool,
because running them one after another would prove nothing about
isolation. Tenant A floods while tenant B sends a normal trickle, and B's
success rate is the number worth looking at.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field

# Stands in for the model call every accepted request has to make.
WORK_SECONDS = 0.01


@dataclass
class TokenBucket:
    """Refills at `rate` tokens a second, never holding more than `burst`."""

    rate: float
    burst: float
    tokens: float = field(init=False)
    updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.burst
        self.updated_at = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated_at) * self.rate)
        self.updated_at = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


@dataclass
class Tally:
    tenant: str
    served: int = 0
    rejected: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.served + self.rejected

    @property
    def success_rate(self) -> float:
        return self.served / self.attempted if self.attempted else 0.0


async def handle(bucket: TokenBucket, tally: Tally, semaphore: asyncio.Semaphore) -> None:
    """One request: check the tenant's bucket, then use shared capacity."""
    if not bucket.take():
        tally.rejected += 1
        return

    started = time.perf_counter()
    async with semaphore:  # the shared resource both tenants compete for
        await asyncio.sleep(WORK_SECONDS)
    tally.served += 1
    tally.latencies.append((time.perf_counter() - started) * 1000)


async def tenant_traffic(
    name: str, requests: int, bucket: TokenBucket, semaphore: asyncio.Semaphore
) -> Tally:
    tally = Tally(name)
    await asyncio.gather(*(handle(bucket, tally, semaphore) for _ in range(requests)))
    return tally


async def run(flood: int, trickle: int, rate: float, burst: float, capacity: int,
              limit_enabled: bool) -> list[Tally]:
    semaphore = asyncio.Semaphore(capacity)

    def bucket_for(_: str) -> TokenBucket:
        if limit_enabled:
            return TokenBucket(rate=rate, burst=burst)
        # A bucket nobody can exhaust, to show what happens without limits.
        return TokenBucket(rate=1e9, burst=1e9)

    return list(
        await asyncio.gather(
            tenant_traffic("A (flooding)", flood, bucket_for("A"), semaphore),
            tenant_traffic("B (normal)", trickle, bucket_for("B"), semaphore),
        )
    )


def report(tallies: list[Tally], limit_enabled: bool) -> None:
    heading = "with a per-tenant limit" if limit_enabled else "with no limit at all"
    print(f"\nBoth tenants running at once, {heading}")
    print(f"{'tenant':<16}{'sent':>7}{'served':>8}{'rejected':>10}{'success':>9}{'p99 ms':>9}")
    print("-" * 62)
    for tally in tallies:
        p99 = 0.0
        if tally.latencies:
            ordered = sorted(tally.latencies)
            p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
        print(
            f"{tally.tenant:<16}{tally.attempted:>7}{tally.served:>8}"
            f"{tally.rejected:>10}{tally.success_rate:>8.0%}{p99:>9.1f}"
        )
    print("-" * 62)

    neighbour = next(t for t in tallies if t.tenant.startswith("B"))
    if limit_enabled:
        print(
            f"B got {neighbour.success_rate:.0%} of its requests through "
            f"at {max(neighbour.latencies, default=0):.0f} ms worst case, "
            "while A was flooding."
        )
        print("A was refused the moment its bucket ran dry, so it never queued.")
    else:
        print("Nothing was refused, so A's flood competed with B for every worker.")
        print("B still succeeds, but look at its p99 against the limited run.")
        print("That latency is the cost of having no limit, paid by the wrong tenant.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flood", type=int, default=400, help="requests from tenant A")
    parser.add_argument("--trickle", type=int, default=20, help="requests from tenant B")
    parser.add_argument("--rate", type=float, default=50.0, help="tokens per second")
    parser.add_argument("--burst", type=float, default=50.0, help="bucket size")
    parser.add_argument("--capacity", type=int, default=8, help="shared workers")
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="remove the buckets, to show what sharing looks like unprotected",
    )
    args = parser.parse_args(argv)

    tallies = asyncio.run(
        run(args.flood, args.trickle, args.rate, args.burst,
            args.capacity, not args.no_limit)
    )
    report(tallies, not args.no_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
