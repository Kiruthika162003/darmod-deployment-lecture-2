# Reliability, maintainability and scalability

Ten small programs from DARMOD Lecture 2. Lecture 1 was about getting a
model into production; this is about what happens to it afterwards.

Every number these print is measured while they run. Nothing is a lookup
table and nothing is a `sleep` pretending to be work. Where a demo cannot
honestly show something on a laptop it says so in its output rather than
inventing a figure — the clearest example is `inference_cost.py`, which
reports half precision as *slower* on CPU, because it is.

Each file is a few hundred lines and stands alone. Read the docstring at
the top of any of them and you have the concept.

## Running them

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r lecture2/requirements.txt
cd lecture2
```

Everything works with no cloud account. Three demos will use Google Cloud
if it is configured, and fall back to a local file if not:

```bash
export GOOGLE_CLOUD_PROJECT=your-project   # Firestore, Pub/Sub, Gemini
export RETRAIN_TOPIC=retrain-topic         # optional, for retraining.py
export REDIS_URL=redis://localhost:6379    # optional, for response_cache.py
```

## Keeping a model honest

**`feature_store.py`** — one write lands in an online store and an offline
log, so training and serving cannot disagree. The `check` command reads
both back and exits 2 when they do.

```bash
python feature_store.py reset u1001            # start clean
python feature_store.py write u1001 risk_score 0.83
python feature_store.py check u1001            # agrees, exit 0
python feature_store.py diverge u1001 risk_score 0.61
python feature_store.py check u1001            # SKEW, exit 2
```

Run `reset` first if you have run this before: the online store outlives
the process, and leftovers from an earlier run will show up as a skew you
did not cause.

**`drift.py`** — PSI and KS, both implemented from scratch. The sample
generator shifts only the age distribution, so a working detector should
flag age and leave the other three columns alone.

```bash
python drift.py sample
python drift.py check yesterday.csv today.csv  # age drifts, exit 1
python drift.py check yesterday.csv yesterday.csv   # exactly zero
```

**`retraining.py`** — a logistic regression trained on week zero and then
scored against cohorts that genuinely move away from it. Accuracy falls
from 0.86 to below 0.80 over six weeks because the data changed, not
because a counter was decremented. Publishes a signal and stops; a worker
subscribing to the topic would do the retraining.

```bash
python retraining.py --floor 0.80
```

## Shipping it safely

**`pipeline.py`** — eight real stages with a drift gate between evaluating
and packaging. There is no stage that pretends to deploy.

```bash
python pipeline.py                # all pass
python pipeline.py --force-drift  # stops at the gate, nothing packaged
```

## Handling load

**`scaling.py`** — vertical, horizontal and async on the same workload.
The worker pool is warmed before timing starts; without that, most of the
horizontal result is the cost of spawning processes.

**`autoscaling.py`** — a real server, a real traffic wave, and replica
counts computed from the rate actually measured. Traffic falls off a
cliff so the slow scale-down is visible rather than merely claimed.

```bash
python scaling.py --jobs 40
python autoscaling.py --duration 12 --peak-rps 40
```

## Paying less

**`batching.py`** — continuous batching reduced to the matrix multiply it
comes down to. Identical arithmetic, one shape against another.

**`inference_cost.py`** — batching, half precision and buffer reuse,
measured independently. Read the half precision row: it halves memory and
runs slower here, which is the honest CPU result.

**`response_cache.py`** — a real Gemini call, then the same question
served from cache. The `templating` command prints the cache key twice a
second apart, with and without a timestamp in the prompt.

```bash
python batching.py
python inference_cost.py
python response_cache.py ask "What is Vector Institute?"
python response_cache.py ask "What is Vector Institute?"   # hit
python response_cache.py templating
```

## Sharing a service

**`multitenant.py`** — a token bucket per tenant, with both tenants
running at once. Run it with `--no-limit` and every request still
succeeds; what degrades is the well behaved tenant's p99, which is the
whole argument for per-tenant limits.

```bash
python multitenant.py
python multitenant.py --no-limit
```

## What these deliberately do not do

`pipeline.py` packages a model but does not deploy it. `retraining.py`
sends a signal but does not retrain. `batching.py` and `inference_cost.py`
run on CPU, so they demonstrate why GPU serving techniques work rather
than benchmarking a GPU.

Each of those is a boundary worth stating out loud when teaching from
them, because the gap between the demo and production is where most of
the real engineering lives.
