"""Watch an autoscaler react to a traffic wave you can actually see.

A real server starts on a free port, a real load generator ramps requests
up and back down, and the replica count is worked out from the request
rate that was measured rather than from a script of what should happen.

The rule is Cloud Run's, and it is simpler than people expect:

    replicas = ceil(requests_in_flight / concurrency)

clamped between --min-instances and --max-instances. Everything else is
consequence. Concurrency of 1 on a GPU model means one replica per request
and a large bill. Concurrency of 80 on an IO-bound HTTP service means one
replica absorbs most of a spike.

Scale up is immediate because a queued request is a user waiting. Scale
down is deliberately slow, because tearing down a replica you are about to
need again costs a cold start. That asymmetry is visible in the output:
the bars climb faster than they fall.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import socket
import sys
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI

# How many replicas we are willing to release per tick on the way down.
SCALE_DOWN_STEP = 1


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/predict")
    async def predict() -> dict[str, float]:
        # Stand-in for model latency. Async, so one worker can hold many.
        await asyncio.sleep(0.02)
        return {"score": 0.83}

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    """A uvicorn server on a background thread, with a readiness wait."""

    def __init__(self, port: int) -> None:
        config = uvicorn.Config(
            build_app(), host="127.0.0.1", port=port, log_level="error"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self) -> Server:
        self._thread.start()
        deadline = time.time() + 15
        while time.time() < deadline:
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start within 15 seconds")

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def wave(second: int, duration: int, peak_rps: int) -> int:
    """Ramp up to the peak, then fall off a cliff.

    The cliff is the interesting part. Traffic vanishes in one second but
    the replica count walks down slowly behind it, which is exactly the
    behaviour you are paying for and the reason a scale-to-zero service
    still bills you for a while after everyone leaves.
    """
    climb = max(1, int(duration * 0.5))
    if second <= climb:
        return max(1, round(peak_rps * second / climb))
    return 1


async def drive(url: str, duration: int, peak_rps: int, concurrency: int,
                min_instances: int, max_instances: int) -> None:
    replicas = min_instances
    print(f"\n{'sec':>4}{'sent':>7}{'ok':>6}{'rps':>8}{'replicas':>10}   load")
    print("-" * 78)

    async with httpx.AsyncClient(timeout=10.0) as client:
        for second in range(1, duration + 1):
            target = wave(second, duration, peak_rps)
            started = time.perf_counter()
            results = await asyncio.gather(
                *(client.get(f"{url}/predict") for _ in range(target)),
                return_exceptions=True,
            )
            elapsed = time.perf_counter() - started

            ok = sum(1 for r in results if not isinstance(r, Exception))
            # Rate over the whole one-second tick, not over the burst. A
            # burst of 13 requests finishing in 45ms is 13 requests that
            # second, not 285 per second.
            measured_rps = ok / max(elapsed, 1.0)

            wanted = max(min_instances, math.ceil(measured_rps / concurrency))
            wanted = min(wanted, max_instances)
            if wanted > replicas:
                replicas = wanted  # up immediately
            elif wanted < replicas:
                replicas = max(wanted, replicas - SCALE_DOWN_STEP)  # down gently

            bar = "#" * replicas
            print(
                f"{second:>4}{target:>7}{ok:>6}{measured_rps:>8.1f}"
                f"{replicas:>10}   {bar}"
            )

            remaining = 1.0 - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

    print("-" * 78)
    print("Replica counts come from the rps actually measured each second.")
    print("Up is immediate. Down releases at most one replica per second.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--duration", type=int, default=12, help="seconds")
    parser.add_argument("--peak-rps", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=10)
    args = parser.parse_args(argv)

    port = free_port()
    print(f"Starting a real server on port {port}")
    with Server(port) as server:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(
                drive(
                    server.url,
                    args.duration,
                    args.peak_rps,
                    args.concurrency,
                    args.min_instances,
                    args.max_instances,
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
