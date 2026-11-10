# Maple

Maple is a small AI agent built for Vector Institute's DARMOD program,
Lecture 1: Model Deployment. She answers questions about Vector Institute,
serves the same underlying agent through three different inference modes
(online, streaming, and a flash-vs-pro comparison), and has one
human-in-the-loop gated action — ordering swag from a little storefront,
which sits in a pending queue until an admin approves it.

The code is small on purpose. Every file is short enough to actually read.

**Lecture 2 (keeping it alive):** open `/lab` and read [LECTURE2.md](LECTURE2.md).

**New here? Start with [COOKBOOK.md](COOKBOOK.md)** — it walks through
running this locally with a free API key, understanding what each part
does, and deploying it to Google Cloud Run, including the two things that
will trip you up if you don't know about them ahead of time.

## Quick start

```bash
git clone <this-repo-url> maple && cd maple
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env   # then add your GOOGLE_API_KEY - free at aistudio.google.com/apikey
uvicorn backend.main:app --reload --port 8080
```

Open `http://localhost:8080`.

## Stack

FastAPI backend, plain HTML/CSS/JS frontend (no build step), Google ADK for
the agent loop, Gemini for the model. Runs locally with a free AI Studio
key, or on Google Cloud Run with a service account and no key in the
container at all — same code, either path.

## Tests

```bash
pytest -q
```

No network calls. Covers the session store, the tools, and the FastAPI
routes.

## License

MIT. Built as a teaching example — take it apart, break it, make it yours.
