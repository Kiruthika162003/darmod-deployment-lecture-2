# The Maple Cookbook

This is the code from Lecture 1 of Vector Institute's DARMOD "Model
Deployment" session. Maple is a small AI agent — a beaver, if you're
curious — that answers questions about Vector Institute, runs in three
different inference modes from the same backend, and has one action gated
behind human approval before it's allowed to run. It's built to be read,
not just run: every file is small enough to sit down and actually
understand in one sitting.

This doc walks you through getting it running on your own machine, then
putting it on the internet, in the order you'll actually hit problems.

## What you end up with

A page that looks like a chat app with four tabs (Online, Streaming,
Compare, All) and a little storefront on the side. Ask it something, it
answers. Try ordering a hoodie from the storefront and watch the order sit
in a pending queue until someone approves it from `/admin` — that's the
human-in-the-loop part.

## Before you start

You need:

- Python 3.12+
- A free Google AI Studio account (no credit card, no GCP project needed)
- 10 minutes

You do *not* need a GCP account for the local version. That's on purpose —
the whole point of the dual-auth setup below is that you can run this
without owning any cloud infrastructure.

## Step 1: get a key

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey),
sign in with any Google account, click "Create API key." Copy it somewhere
you can paste from in a minute. This key is free up to a generous daily
quota — you will not hit it just poking around.

## Step 2: clone and set up a virtual environment

```bash
git clone <this-repo-url> maple
cd maple

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
```

That last command takes a minute or two — it's pulling in `google-adk`,
FastAPI, and their dependencies.

## Step 3: configure

```bash
cp .env.example .env
```

Open `.env` and uncomment/fill in one line:

```
GOOGLE_API_KEY=AIzaYOUR_KEY_HERE
```

Everything else in `.env.example` already has a sane default. You don't
need to touch anything else to get this running.

## Step 4: run it

```bash
uvicorn backend.main:app --reload --port 8080
```

Open `http://localhost:8080`. Type something into the Online tab — try
"What is Vector Institute?" — and hit Ask. You should get a real answer in
a couple seconds.

If instead you see an error in the reply bubble, read it — it's an actual
error message now, not a silent blank. The most common one at this step is
a malformed key (copy-paste dropped a character) or the key not being
saved because `.env` wasn't actually in the folder you ran `uvicorn` from.

## What's actually happening when you click Ask

The tabs are five ways of calling the same agent, and looking at them side
by side is the fastest way to understand the difference between inference
modes:

- **Online** (`POST /chat`) — one request, one full reply. You wait, you
  get the whole answer at once. This is `backend/main.py`'s `chat()`
  function, about ten lines.
- **Streaming** (`POST /chat/stream`) — same question, but the response
  comes back as Server-Sent Events, one token at a time, so the reply
  appears word by word instead of all at once. Same underlying call to the
  agent, different response shape.
- **Batch** (`POST /chat/batch`) — many prompts in one request, many
  replies back together. Nobody waits at the screen; this is the shape you
  use for work that can run overnight.
- **Compare** — fires the same prompt at the chat model and the Pro model
  at the same time. It shows latency, the tokens each model reports
  (including the thinking tokens), and the cost those tokens add up to.
- **Test Lab** — runs one prompt in online, streaming and batch at once so
  you can watch them race.

None of these are separate models or separate agents. It's one agent
(`backend/agent.py`), called different ways. That's the thing worth
noticing: inference mode is a property of how you serve a model, not a
property of the model itself.

Under every Online and Streaming reply there is a list of **steps**: what
Maple did to produce it. For a factual question that is usually "decide to
call `search_agent`", "`search_agent` returns", "write the answer", each
with the time it took, plus the model calls and tokens used. Most of the
wait is the search, not the answer. Untick **Search the web before
answering** and ask again to see the difference: a reply can drop from several
seconds to one or two, but Maple then answers from memory and can get
Vector facts wrong. That trade-off, speed against accuracy, is a decision
you make every time an agent is allowed to use a tool.

Each reply also says whether it was a **cold start**: the first request a
new instance serves, which has to build the agent and the model client
while you wait. On Cloud Run with `--min-instances 0` you will see it after
the service has been idle.

### Models, thinking and regions

`CHAT_MODEL` defaults to `gemini-3.1-flash-lite` and `PRO_MODEL` to
`gemini-3.1-pro-preview`. `THINKING_BUDGET=0` switches thinking off for the
chat model, which is what keeps replies fast: with thinking on, Gemini 2.5
Flash spent about 285 thinking tokens on a ten-token answer. Pro cannot
switch thinking off, which is a large part of why it is slower and costs
more.

Both default models are served from the `global` endpoint only. They are
not available in Montreal, Toronto or even `us-central1`, so prompts may be
processed outside Canada. If data has to stay in Canada, use
`gemini-2.5-flash` and `gemini-2.5-pro` with `northamerica-northeast1`
(Montreal); Toronto (`northamerica-northeast2`) serves none of them. Pick the
model and the region together, and say out loud which one you picked.

The storefront on the right is the human-in-the-loop demo. Ask Maple to
order you a hoodie, or click an item and fill in your name and email. The
agent calls a tool named `submit_order`, but that tool is wrapped in a
gate (`backend/agent.py`, `_maple_before_tool_callback`) that intercepts
it before it runs, parks it in a queue, and tells the model "this is
pending human approval" instead of letting it complete. Go to
`/admin` (password `1234` unless you changed `ADMIN_PASSWORD`), find your
order, click Approve, and watch the chat update. That one function is the
entire mechanism — no framework magic, just a callback that can say "not
yet."

## Running the tests

```bash
pytest -q
```

Twenty-ish tests, all fast, none of them call Google (they mock the parts
that would). If you're changing something and want to know you didn't
break the plumbing, this is the fast feedback loop — the slow one is
clicking through the UI by hand.

## Step 5: put it in a container

```bash
docker build -t maple:v1 .
docker images maple:v1
```

Check the size. It should land around 250MB. The `Dockerfile` here is
multi-stage on purpose — it builds Python wheels in one stage, then copies
only the built wheels (not the compilers used to build them) into a slim
runtime stage. A naive single-stage build of the same app comes out around
900MB. If you're curious what that habit is worth, comment out the second
`FROM` line and rebuild — you'll see the difference directly.

Run it locally in the container, same key as before:

```bash
docker run --rm -p 8080:8080 \
  -e GOOGLE_API_KEY=AIzaYOUR_KEY_HERE \
  maple:v1
```

Same app, same behavior, now running from a frozen image instead of your
live filesystem. That's the whole point of a container — this exact image
is what goes to the cloud next.

## Step 6: put it on the internet

This part needs an actual GCP project, because Cloud Run is a real Google
Cloud service. If you don't have one, create one at
[console.cloud.google.com](https://console.cloud.google.com) — the free
tier covers this comfortably for a demo.

```bash
gcloud auth login
gcloud auth application-default login

PROJECT=your-gcp-project-id ./deploy.sh
```

That script does four things: turns on the four GCP APIs it needs, builds
your container via Cloud Build, deploys it to Cloud Run, and prints the
URL when it's done. Takes two to four minutes, mostly waiting on the
build.

A couple of things worth knowing before you run this, because they cost me
actual debugging time:

**Don't check "is it up" with `curl $URL/healthz`.** Cloud Run's edge
layer reserves that exact path for its own internal use on the default
`*.run.app` domain, and will 404 it even when your app is completely
healthy. Use `curl $URL/catalog` or just open the URL in a browser
instead. `/healthz` still works from *inside* your own network calls, it's
just not reachable from the public internet on this domain.

**Gemini isn't available in every Vertex AI region.** The script deploys
your Cloud Run container to Toronto (`northamerica-northeast2`) by
default, but points the actual model calls at Montreal
(`northamerica-northeast1`) — because as of this writing, Gemini on Vertex
AI doesn't serve from Toronto yet. If you pick your own region and start
seeing `400 FAILED_PRECONDITION` from the model, this is almost always
why. Check
[Vertex AI's generative AI locations list](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations)
and set `VERTEX_LOCATION` to somewhere that's actually on it — `us-central1`
always works if you just want to confirm it's not your setup.

```bash
VERTEX_LOCATION=us-central1 PROJECT=your-gcp-project-id ./deploy.sh
```

Once it's live, hit the URL from your phone. That's the actual moment this
is trying to prove: the thing on your laptop and the thing on the internet
are now the same artifact.

## Rolling back

Cloud Run keeps every revision. If a deploy goes wrong:

```bash
gcloud run revisions list --service maple --region northamerica-northeast2
gcloud run services update-traffic maple \
  --region northamerica-northeast2 \
  --to-revisions=<previous-revision-name>=100
```

Takes about 30 seconds. No rebuild needed — you're just pointing traffic
at an image that already exists.

## If you want to change something

A few starting points, roughly easiest to hardest:

- **Change the catalog.** `backend/tools.py`, the `CATALOG` dict. Add an
  item, it shows up in the storefront and the agent can sell it.
- **Add a second HITL-gated action.** Write a function like
  `submit_order`, add its name to `hitl_tools` in `backend/config.py`, wire
  it into `backend/agent.py`'s tool list. The approval-queue plumbing
  already handles anything you add this way.
- **Swap the search provider.** `backend/tools.py`'s `web_search` tries
  Google Custom Search, then Tavily, then DuckDuckGo with no key at all.
  Set `TAVILY_API_KEY` in `.env` for noticeably better results than the
  no-key fallback.
- **Point it at a different model.** `CHAT_MODEL` in `.env`. Anything
  Gemini that supports function calling works.

## Where things live

```
backend/
  main.py            FastAPI routes - this is where /chat, /chat/stream,
                      /admin/orders, etc. are defined
  agent.py            The agent itself: system prompt, tools, the HITL gate
  tools.py            web_search and submit_order - what the agent can do
  config.py           All the settings, with defaults, in one place
  session_store.py    Who's talking to Maple, kept in memory
  order_store.py      Pending/approved/denied orders, kept in memory

frontend/
  index.html          The four-tab chat page and storefront
  app.js              Talks to the backend, renders replies, no framework
  admin.html + admin.js   The approval queue page

tests/                Fast, no network calls
Dockerfile             Multi-stage build, non-root user
deploy.sh               One command, Cloud Run
```

Both `session_store.py` and `order_store.py` keep everything in memory —
restart the process and it's gone. That's fine for a demo running on one
instance. If you were taking this further, that's the first thing to swap
for something like Redis, so it survives a restart and works across more
than one instance.
