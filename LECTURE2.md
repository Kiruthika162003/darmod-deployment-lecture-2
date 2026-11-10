# Lecture 2: keeping an AI app alive

Lecture 1 put Maple on the internet. Lecture 2 is about what happens next: more
people arrive, the world changes under the model, and something has to notice.

Open the **lab** at `/lab` (on your laptop `http://localhost:8080/lab`, or add
`/lab` to the live Cloud Run link). It has ten demos, one at a time, and each
one does real work on the server.

| # | Demo | What you do | What it shows |
|---|------|-------------|---------------|
| 1 | One rule, two stores | Write, change the rule, check | A feature store keeps the live app and the model on the same numbers |
| 2 | The people changed | Drag the slider | Data drift: the crowd moves, the model does not |
| 3 | A model gets stale | Press Next week | A real model slips week by week until it is retrained |
| 4 | A gate before shipping | Run a normal day, then a bad-data day | Checks run for real, and bad data stops the release |
| 5 | One, many, or a queue | Start all three lanes | More workers, or a queue that hands back a ticket |
| 6 | Copies appear when busy | Send 60 visitors | On Cloud Run, new copies start; locally there is only one |
| 7 | Answer people together | Run the race | Reading the model once for everyone: the idea behind vLLM |
| 8 | A smaller brain, same answers? | Pick 8, 4, 3, 2 bits | Quantization: models survive shrinking, until they do not |
| 9 | Remember answers | Ask, ask again, reword it | A cache that also matches questions worded differently |
| 10 | Share fairly | Team A floods the service | Give each team its own allowance |

The code is in `backend/lab.py` (every endpoint is under `/lab/api`) and
`frontend/lab.js`. The older scripts from the first version of this lecture are
still in `lecture2/`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn backend.main:app --port 8080
```

You need a Google Cloud login for the cache demo (it calls Gemini and an
embedding model). Everything else runs without one. Set `LAB_FAKE=1` to
rehearse offline with scripted answers.

## Things worth knowing

- **Autoscaling is real only on Cloud Run.** The server sends the crowd to its
  own public address, so Cloud Run spreads it over copies. A browser cannot do
  this: it keeps only a few connections open. Deploy with `--concurrency 8`.
- **The cache lives in each copy's own memory.** With several copies running,
  the same question can be fresh on one and remembered on another. That is the
  reason real systems use a shared store.
- **The cache's cutoff was measured.** With the embedding model, reworded
  questions scored 0.91 or more and different questions about the same topic
  scored 0.87 or less, so a match needs 0.90. A missed match only costs a fresh
  answer; a wrong match hands someone the wrong answer.
- **The batching race caps the maths library at two threads.** Left alone it
  spreads a small job over every core the machine reports, which is slower than
  not spreading it.
