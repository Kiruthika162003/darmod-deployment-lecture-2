/* Lecture 2 lab: ten live demos of what keeps an AI app alive in production.

Each demo talks to /lab/api and does real work on the server. The page only
draws what comes back, one demo at a time, with the steps in order.
*/

(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function h(tag, props = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(kid));
    }
    return el;
  }

  async function api(path, method = "GET", body) {
    const res = await fetch("/lab/api" + path, {
      method,
      credentials: "same-origin",
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function stream(path, onEvent) {
    const res = await fetch("/lab/api" + path, { credentials: "same-origin" });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (block.startsWith("data:")) onEvent(JSON.parse(block.slice(5)));
      }
    }
  }

  const failed = (where) => where.replaceChildren(h("div", { class: "banner bad" }, "Could not reach the server. Is Maple still running?"));

  // 1. Feature store -----------------------------------------------------------
  function featureStore(body) {
    const ruleText = (r) => (r === "running" ? "age = running year" : "age = completed years");
    const value = (id) => h("div", { class: "big", id }, "–");
    const tag = (id) => h("span", { class: "tag", id }, "empty");
    const result = h("div");
    const btn = (n, label, fn, ghost) => h("button", { class: "btn" + (ghost ? " ghost" : ""), onclick: fn }, h("span", { class: "step-no" }, n), label);

    body.append(
      h("div", { class: "cards" },
        h("div", { class: "card" }, h("h3", {}, "Live app reads (online)"), value("fsOn"), tag("fsOnTag")),
        h("div", { class: "card" }, h("h3", {}, "Model trained on (offline)"), value("fsOff"), tag("fsOffTag")),
      ),
      h("div", { class: "row" },
        btn("1", "Write the feature", () => run("/features/write", "POST", 1), false),
        btn("2", "A developer changes the rule", () => run("/features/change-rule", "POST", 2), true),
        btn("3", "Run the live pipeline", () => run("/features/serve", "POST", 3), true),
        btn("4", "Check", check, true),
        h("button", { class: "btn ghost", onclick: reset, style: "margin-left:auto" }, "Reset"),
      ),
      result,
    );
    const buttons = [...body.querySelectorAll(".row .btn")];
    const unlock = (n) => buttons.forEach((b, i) => { if (i < 4) b.disabled = i > n; });
    unlock(0);

    function paint(v) {
      $("#fsOn", body).textContent = v.online ? v.online.age : "–";
      $("#fsOnTag", body).textContent = v.online ? ruleText(v.online.rule) : "empty";
      $("#fsOff", body).textContent = v.offline ? v.offline.age : "–";
      $("#fsOffTag", body).textContent = v.offline ? ruleText(v.offline.rule) : "empty";
    }
    async function run(path, method, step) {
      try {
        paint(await api(path, method));
        result.replaceChildren(
          step === 1 ? h("div", { class: "banner info" }, "Written once, to both stores. Same rule, same number.")
          : step === 2 ? h("div", { class: "banner info" }, "Only the live pipeline has the new rule. The training history does not.")
          : h("div", { class: "banner info" }, "The live app now reads a different number from the one the model learned."),
        );
        unlock(step);
      } catch { failed(result); }
    }
    async function check() {
      try {
        const v = await api("/features/check");
        paint(v);
        result.replaceChildren(v.agree
          ? h("div", { class: "banner ok" }, `In sync: both stores say ${v.online.age}.`)
          : h("div", { class: "banner bad" }, `Skew: the live app sees ${v.online.age}, but the model learned from ${v.offline.age}. Nothing crashed. The answers are just quietly wrong.`));
      } catch { failed(result); }
    }
    async function reset() {
      try { paint(await api("/features/reset", "POST")); result.replaceChildren(); unlock(0); } catch { failed(result); }
    }
    api("/features/reset", "POST").then(paint).catch(() => failed(result));
  }

  // 2. Data drift --------------------------------------------------------------
  function drift(body) {
    const hist = h("div", { class: "hist" });
    const light = h("span", { class: "light" }, "stable");
    const range = h("input", { type: "range", class: "range", min: -25, max: 25, value: 0, step: 1 });
    const label = h("div", { class: "muted" }, "Same crowd as yesterday");
    body.append(
      h("div", { class: "row" }, light, label),
      hist,
      h("div", { class: "legend" }, h("span", {}, h("i", { style: "border:2px solid var(--mid)" }), "Yesterday's applicants"), h("span", {}, h("i", { style: "background:var(--pink)" }), "Today's applicants")),
      h("div", { class: "card" }, h("h3", {}, "Drag: today's applicants get older or younger"), range),
    );
    let timer;
    async function paint() {
      try {
        const v = await api(`/drift?shift=${range.value}`);
        const top = Math.max(...v.yesterday, ...v.today);
        hist.replaceChildren(...v.bins.map((b, i) => h("div", { class: "hist-col" },
          h("div", { class: "hist-bar was", style: `height:${(v.yesterday[i] / top) * 100}%` }),
          h("div", { class: "hist-bar now", style: `height:${(v.today[i] / top) * 100}%` }),
          h("span", { class: "hist-label" }, b))));
        light.className = "light " + { stable: "green", watch: "amber", drifted: "red" }[v.status];
        light.textContent = { stable: "Stable", watch: "Watch closely", drifted: "Drifted: the model has not seen these people" }[v.status];
        const s = Number(range.value);
        label.textContent = s === 0 ? "Same crowd as yesterday" : `Today's applicants are ${Math.abs(s)} years ${s > 0 ? "older" : "younger"}`;
      } catch { failed(hist); }
    }
    range.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(paint, 80); });
    paint();
  }

  // 3. Model decay ------------------------------------------------------------
  function decay(body) {
    const weeks = h("div", { class: "weeks" });
    const light = h("span", { class: "light" }, "Getting ready…");
    const note = h("div", { class: "muted" }, "Training the first model");
    const next = h("button", { class: "btn" }, "Next week");
    const retrain = h("button", { class: "btn ghost" }, "Retrain on this week");
    const reset = h("button", { class: "btn ghost", style: "margin-left:auto" }, "Reset");
    body.append(h("div", { class: "row" }, light, note), h("div", { class: "weeks-wrap" }, weeks), h("div", { class: "row" }, next, retrain, reset));
    const words = { healthy: ["green", "Healthy"], watch: ["amber", "Watch: it is slipping"], retrain: ["red", "Time to retrain"] };
    function paint(v) {
      weeks.replaceChildren(...v.history.map((e) => {
        const cls = e.retrained ? "retrained" : e.verdict;
        const hgt = Math.max(8, ((e.accuracy - 0.5) / 0.5) * 100);
        return h("div", { class: "week", title: `Week ${e.week}: right ${Math.round(e.accuracy * 100)}% of the time` },
          h("div", { class: "week-bar " + cls, style: `height:${hgt}%` }), h("span", {}, e.retrained ? "↻" : `W${e.week}`));
      }));
      const last = v.history[v.history.length - 1];
      light.className = "light " + words[last.verdict][0];
      light.textContent = words[last.verdict][1];
      note.textContent = last.retrained ? `Week ${v.week}: retrained on recent applicants` : `Week ${v.week}: the model is the one trained in week 0`;
      next.disabled = v.week >= 11;
      retrain.disabled = false;
    }
    next.disabled = retrain.disabled = true;  // wait for the first reset to land
    const call = (p) => api(p, "POST").then(paint).catch(() => failed(weeks));
    next.onclick = () => call("/decay/next");
    retrain.onclick = () => call("/decay/retrain");
    reset.onclick = () => call("/decay/reset");
    call("/decay/reset");
  }

  // 4. Pipeline with a data gate ----------------------------------------------
  function pipeline(body) {
    const names = ["Code check", "Unit checks", "Train", "Evaluate", "Data gate", "Register"];
    const cards = new Map(names.map((n) => [n, h("div", { class: "stage-card" }, h("span", { class: "mark" }, ""), h("b", {}, n), h("small", {}, "waiting"))]));
    const banner = h("div");
    const go = (day) => async () => {
      buttons.forEach((b) => (b.disabled = true));
      banner.replaceChildren();
      cards.forEach((c) => { c.className = "stage-card"; c.children[0].replaceChildren(); c.children[2].textContent = "waiting"; });
      let stopped = null;
      try {
        await stream(`/pipeline/run?day=${day}`, (e) => {
          const c = cards.get(e.stage);
          if (!c) return;
          c.className = "stage-card " + e.status;
          c.children[0].replaceChildren(e.status === "running" ? h("span", { class: "spinner" }) : document.createTextNode({ pass: "✓", fail: "✕", skipped: "–" }[e.status] || ""));
          c.children[2].textContent = e.note || { running: "running…", skipped: "skipped" }[e.status] || "waiting";
          if (e.status === "fail") stopped = e;
        });
        banner.replaceChildren(stopped
          ? h("div", { class: "banner bad" }, "Stopped at the gate. Bad data never reaches a customer.")
          : h("div", { class: "banner ok" }, "Every check passed. This version is ready to ship."));
      } catch { failed(banner); }
      buttons.forEach((b) => (b.disabled = false));
    };
    const buttons = [
      h("button", { class: "btn", onclick: go("normal") }, "Run on a normal day"),
      h("button", { class: "btn danger", onclick: go("bad") }, "Run on a bad-data day"),
    ];
    body.append(h("div", { class: "pipeline" }, [...cards.values()]), h("div", { class: "row" }, buttons), banner);
  }

  // 5. Scaling shapes ---------------------------------------------------------
  function shapes(body) {
    const JOBS = 12;
    const lanes = [
      { mode: "one", name: "One worker", sub: "one job at a time" },
      { mode: "many", name: "Four workers", sub: "more copies side by side" },
      { mode: "queue", name: "Queue and workers", sub: "take a ticket, come back later" },
    ].map((l) => ({ ...l, dots: h("div", { class: "dots" }), timer: h("div", { class: "timer" }, "0.0 s"), ticket: h("div", { class: "ticket" }) }));
    const draw = (l) => l.dots.replaceChildren(...Array.from({ length: JOBS }, () => h("div", { class: "dot" })));
    lanes.forEach(draw);
    const start = h("button", { class: "btn" }, "Start all three");
    body.append(
      h("div", { class: "lanes" }, lanes.map((l) => h("div", { class: "lane" }, h("div", {}, h("b", {}, l.name), h("small", {}, l.sub)), h("div", {}, l.dots, l.ticket), l.timer))),
      h("div", { class: "row" }, start, h("span", { class: "muted" }, "Each job waits 0.4 s, like a call to a model. Twelve jobs in each lane.")),
    );
    start.onclick = async () => {
      start.disabled = true;
      lanes.forEach((l) => { draw(l); l.ticket.textContent = ""; l.timer.textContent = "0.0 s"; });
      try {
        await Promise.all(lanes.map((l) => {
          let n = 0;
          return stream(`/scale/run?mode=${l.mode}&jobs=${JOBS}`, (e) => {
            l.timer.textContent = (e.ms / 1000).toFixed(1) + " s";
            if (e.kind === "ticket") l.ticket.textContent = `Ticket ${e.ticket} handed over at once. You are free.`;
            if (e.kind === "job") l.dots.children[n++]?.classList.add("done");
          });
        }));
      } catch { failed(body.querySelector(".lanes")); }
      start.disabled = false;
    };
  }

  // 6. Autoscaling ------------------------------------------------------------
  function copies(body) {
    let visitors = 30;
    const box = h("div", { class: "copies" });
    const status = h("div", { class: "muted" }, "Send some visitors to see which copies answer.");
    const seen = new Map();
    const chips = [10, 30, 60].map((n) => h("button", { class: "chip" + (n === visitors ? " on" : ""), onclick: () => { visitors = n; chips.forEach((c) => c.classList.toggle("on", c === chips[[10, 30, 60].indexOf(n)])); } }, `${n} visitors`));
    const send = h("button", { class: "btn" }, "Send them now");
    body.append(h("div", { class: "row" }, chips, send), box, status);
    send.onclick = async () => {
      send.disabled = true;
      seen.clear();
      box.replaceChildren();
      let answered = 0;
      status.textContent = `${visitors} visitors on their way…`;
      try {
        await stream(`/scale/crowd?n=${visitors}`, (e) => {
          if (e.kind !== "visitor") return;
          if (!seen.has(e.copy)) {
            const num = h("div", { class: "big" }, "0");
            seen.set(e.copy, { n: 0, num });
            box.append(h("div", { class: "copy-box" }, h("b", {}, "copy " + e.copy), num, h("small", {}, "visitors answered")));
          }
          const c = seen.get(e.copy);
          c.num.textContent = ++c.n;
          status.textContent = `${visitors - ++answered} visitors still waiting…`;
        });
      } catch { failed(status); send.disabled = false; return; }
      status.textContent = seen.size === 1
        ? "One copy answered everyone. On Cloud Run, a crowd makes more copies appear."
        : `${seen.size} copies shared the crowd. Cloud Run started them when the first one got busy.`;
      send.disabled = false;
    };
  }

  // 7. Serving many at once ---------------------------------------------------
  function together(body) {
    const fill = (cls) => h("div", { class: "race-fill " + cls });
    const a = fill(""), b = fill("fast");
    const ta = h("div", { class: "timer" }, ""), tb = h("div", { class: "timer" }, "");
    const verdict = h("div");
    const go = h("button", { class: "btn" }, "Run the race");
    body.append(
      h("div", { class: "race" },
        h("div", { class: "race-row" }, h("div", {}, h("b", {}, "One person at a time"), h("div", { class: "muted" }, "read the model for each")), h("div", { class: "race-track" }, a), ta),
        h("div", { class: "race-row" }, h("div", {}, h("b", {}, "Everyone together"), h("div", { class: "muted" }, "read the model once")), h("div", { class: "race-track" }, b), tb)),
      h("div", { class: "row" }, go, h("span", { class: "muted" }, "64 people, the same maths, run right now on this computer.")),
      verdict,
    );
    go.onclick = async () => {
      go.disabled = true;
      a.style.width = b.style.width = "0";
      ta.textContent = tb.textContent = "";
      verdict.replaceChildren();
      try {
        const r = await api("/batch/run");
        await sleep(60);
        a.style.width = "100%";
        b.style.width = Math.max(4, (r.together_ms / r.one_by_one_ms) * 100) + "%";
        ta.textContent = r.one_by_one_ms + " ms";
        tb.textContent = r.together_ms + " ms";
        verdict.replaceChildren(h("div", { class: "banner ok" }, `About ${Math.max(1, Math.round(r.one_by_one_ms / r.together_ms))}× quicker for the same answers. That is what serving engines like vLLM do for many users at once.`));
      } catch { failed(verdict); }
      go.disabled = false;
    };
  }

  // 8. Shrinking the model ----------------------------------------------------
  function shrink(body) {
    let sample = 1, bits = 32, data = null;  // a digit that survives 3 bits and fails at 2
    const digit = h("div", { class: "digit" });
    const guess = h("div", { class: "guess" }, "?");
    const verdict = h("div", { class: "light" });
    const size = h("div", { class: "size-fill" });
    const accuracy = h("div", { class: "muted" });
    const chips = [32, 16, 8, 4, 3, 2].map((n) => h("button", { class: "chip" + (n === bits ? " on" : ""), onclick: () => { bits = n; chips.forEach((c, i) => c.classList.toggle("on", [32, 16, 8, 4, 3, 2][i] === n)); paint(); } }, n === 32 ? "full" : n + "-bit"));
    const next = h("button", { class: "btn ghost" }, "Another digit");
    body.append(h("div", { class: "quant" },
      h("div", {}, digit, h("div", { class: "muted", style: "margin-top:8px" }, "A handwritten digit the model must read")),
      h("div", { style: "display:grid;gap:16px" },
        h("div", { class: "row" }, chips, next),
        h("div", { class: "row", style: "gap:22px" }, h("div", {}, h("div", { class: "muted" }, "The model reads it as"), guess), verdict),
        h("div", {}, h("div", { class: "muted" }, "How much room the model takes"), h("div", { class: "size-track" }, size)),
        accuracy)));
    function paint() {
      if (!data) return;
      const row = data.rows.find((r) => r.bits === bits);
      digit.replaceChildren(...data.pixels.flat().map((p) => h("i", { style: `background:rgb(${255 - p * 16},${255 - p * 16},${255 - p * 16})` })));
      guess.textContent = row.guess;
      const ok = row.guess === data.truth;
      verdict.className = "light " + (ok ? "green" : "red");
      verdict.textContent = ok ? "Right" : `Wrong: it is a ${data.truth}`;
      size.style.width = row.size * 100 + "%";
      accuracy.textContent = `Across hundreds of digits it is right ${Math.round(row.accuracy * 100)}% of the time at this size.`;
    }
    async function load() { try { data = await api(`/quant?sample=${sample}`); paint(); } catch { failed(accuracy); } }
    next.onclick = () => { sample += 7; load(); };
    load();
  }

  // 9. Cache ------------------------------------------------------------------
  function cache(body) {
    const input = h("input", { type: "text", placeholder: "Ask Maple anything about Vector Institute", autocomplete: "off" });
    const ask = h("button", { class: "btn" }, "Ask");
    const out = h("div");
    const history = h("div", { class: "history" });
    const forget = h("button", { class: "btn ghost" }, "Forget everything");
    const examples = ["What is DARMOD?", "what's darmod", "Where is Vector Institute based?"].map((q) => h("button", { class: "chip", onclick: () => { input.value = q; go(); } }, q));
    body.append(h("div", { class: "ask" }, input, ask), h("div", { class: "row" }, h("span", { class: "muted" }, "Try:"), examples, forget), out, history);
    const label = { fresh: "Fresh from Gemini", exact: "Remembered", similar: "Remembered a similar question" };
    async function go() {
      const q = input.value.trim();
      if (!q) return;
      ask.disabled = true;
      const t0 = performance.now();
      const tick = setInterval(() => out.replaceChildren(h("div", { class: "answer" }, h("span", { class: "source fresh" }, `Asking Gemini… ${((performance.now() - t0) / 1000).toFixed(1)} s`))), 100);
      try {
        const r = await api("/cache/ask", "POST", { question: q });
        clearInterval(tick);
        const took = r.ms < 50 ? "instant" : (r.ms / 1000).toFixed(1) + " s";
        out.replaceChildren(h("div", { class: "answer" },
          h("span", { class: "source " + r.source }, `${label[r.source]} · ${took}`),
          r.matched ? h("div", { class: "muted", style: "margin-top:8px" }, `It matched: “${r.matched}”`) : null,
          h("p", {}, r.answer),
          h("div", { class: "muted", style: "margin-top:8px" }, `Answered by copy ${r.copy}`)));
        history.prepend(h("div", {}, h("span", { class: "src source " + r.source }, r.source === "fresh" ? "fresh" : "remembered"), q, h("span", { class: "muted" }, took)));
      } catch { clearInterval(tick); failed(out); }
      ask.disabled = false;
    }
    ask.onclick = go;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    forget.onclick = async () => { await api("/cache/clear", "POST").catch(() => {}); out.replaceChildren(); history.replaceChildren(); };
  }

  // 10. Fair use --------------------------------------------------------------
  function teams(body) {
    let fair = true;
    const lane = (name) => ({ name, dots: h("div", { class: "dots" }), sum: h("div", { class: "muted" }) });
    const A = lane("Team A"), B = lane("Team B");
    const mode = [
      h("button", { class: "chip on", onclick: () => setFair(true) }, "Each team has its own allowance"),
      h("button", { class: "chip", onclick: () => setFair(false) }, "One shared allowance for everyone"),
    ];
    const flood = h("button", { class: "btn danger" }, "Team A floods the service");
    const normal = h("button", { class: "btn" }, "Team B sends 5 requests");
    const note = h("div");
    const row = (l, sub) => h("div", { class: "team" }, h("div", {}, h("b", {}, l.name), h("div", { class: "muted" }, sub)), l.dots, l.sum);
    body.append(h("div", { class: "row" }, mode), h("div", { class: "lanes" }, row(A, "sends 30 at once"), row(B, "sends 5, politely")), h("div", { class: "row" }, flood, normal), note);
    function setFair(f) { fair = f; mode.forEach((c, i) => c.classList.toggle("on", (i === 0) === f)); api("/tenant/reset", "POST").catch(() => {}); }
    async function fire(l, n, gap) {
      l.dots.replaceChildren();
      let ok = 0;
      for (let i = 0; i < n; i++) {
        const r = await api(`/tenant/request?team=${l.name.slice(-1)}&fair=${fair}`, "POST");
        ok += r.accepted ? 1 : 0;
        l.dots.append(h("div", { class: "dot " + (r.accepted ? "ok" : "no") }));
        l.sum.textContent = `${ok} served, ${i + 1 - ok} told to slow down`;
        if (gap) await sleep(gap);
      }
      return ok;
    }
    flood.onclick = async () => { flood.disabled = true; try { await fire(A, 30, 25); } catch { failed(note); } flood.disabled = false; };
    normal.onclick = async () => {
      normal.disabled = true;
      try {
        const ok = await fire(B, 5, 120);
        note.replaceChildren(ok === 5
          ? h("div", { class: "banner ok" }, "Team B was served in full. One noisy neighbour did not become everyone's bad day.")
          : h("div", { class: "banner bad" }, `Team B got only ${ok} of 5, because Team A used up the shared allowance.`));
      } catch { failed(note); }
      normal.disabled = false;
    };
    api("/tenant/reset", "POST").catch(() => {});
  }

  // Panels ---------------------------------------------------------------------
  const PANELS = [
    { name: "Feature store", sub: "one rule, two stores", title: "One rule, two stores", idea: "The live app and the model must read the same numbers.", say: "<b>Say:</b> nothing crashed, yet the model is now quietly wrong. That is why teams keep one definition for both.", render: featureStore },
    { name: "Data drift", sub: "the people changed", title: "The people changed", idea: "The model has not changed. Its customers have.", say: "<b>Say:</b> a drifted crowd looks normal on every dashboard. You have to compare today with what the model learned from.", render: drift },
    { name: "Model decay", sub: "stale by degrees", title: "A model gets stale", idea: "Nobody touches it, and it still gets worse.", say: "<b>Say:</b> it never breaks loudly. It slips a little each week until someone retrains it.", render: decay },
    { name: "Safe shipping", sub: "a gate before release", title: "A gate before shipping", idea: "Checks run for real. Bad data stops the release.", say: "<b>Say:</b> the code was fine and the model was fine. Only the data was wrong, so the gate is what protects customers.", render: pipeline },
    { name: "Scaling shapes", sub: "one, many, queue", title: "One, many, or a queue", idea: "Three ways to handle more work.", say: "<b>Say:</b> more workers finish sooner. A queue gives the person a ticket so nobody waits at the counter.", render: shapes },
    { name: "Autoscaling", sub: "copies appear", title: "Copies appear when busy", idea: "Send a crowd and see who answers.", say: "<b>Say:</b> on Cloud Run each copy handles a few visitors, so a crowd makes new copies start. At zero visitors there are none.", render: copies },
    { name: "Serving together", sub: "the idea behind vLLM", title: "Answer people together", idea: "Real maths, run right now.", say: "<b>Say:</b> reading the model once for everyone is far cheaper than once per person. Engines like vLLM are built around this.", render: together },
    { name: "Shrinking a model", sub: "smaller, same answers?", title: "A smaller brain, same answers?", idea: "Squeeze the model and watch what it can still read.", say: "<b>Say:</b> models survive a lot of shrinking, until suddenly they do not. That is quantization.", render: shrink },
    { name: "Cache", sub: "remember answers", title: "Remember answers", idea: "Ask twice. Then ask it differently.", say: "<b>Say:</b> the same question costs a second call and a few seconds again. A cache answers instantly, even when it is worded differently.", render: cache },
    { name: "Fair use", sub: "many teams, one service", title: "Share fairly", idea: "One team gets noisy. Does everyone suffer?", say: "<b>Say:</b> give every team its own allowance and one customer's bad day stays theirs.", render: teams },
  ];

  const rail = $("#rail");
  const stage = $("#stage");
  let current = -1;

  PANELS.forEach((p, i) => {
    rail.append(h("button", { class: "rail-item", "data-i": i, onclick: () => show(i) },
      h("span", { class: "rail-num" }, i + 1), h("span", { class: "rail-name" }, p.name, h("span", { class: "rail-sub" }, p.sub))));
  });

  function show(i) {
    if (i < 0 || i >= PANELS.length) return;
    current = i;
    const p = PANELS[i];
    rail.querySelectorAll(".rail-item").forEach((el, k) => el.classList.toggle("active", k === i));
    const bodyEl = h("div", { class: "stage-body" });
    const say = h("div", { class: "stage-say" });
    say.innerHTML = p.say;
    stage.replaceChildren(
      h("div", { class: "stage-head" }, h("span", { class: "stage-num" }, `${i + 1} of ${PANELS.length}`), h("h1", {}, p.title)),
      h("div", { class: "stage-idea" }, p.idea),
      bodyEl,
      say,
      h("div", { class: "stage-nav" },
        h("button", { class: "btn ghost", onclick: () => show(i - 1), disabled: i === 0 ? "" : null }, "← Back"),
        h("button", { class: "btn ghost", onclick: () => show(i + 1), disabled: i === PANELS.length - 1 ? "" : null }, "Next →")),
    );
    history.replaceState(null, "", "#" + (i + 1));
    p.render(bodyEl);
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowRight") show(current + 1);
    if (e.key === "ArrowLeft") show(current - 1);
  });

  api("/whoami").then((w) => { $("#copyChip").textContent = "copy " + w.copy; }).catch(() => { $("#copyChip").textContent = "offline"; });
  const start = parseInt(location.hash.slice(1), 10);
  show(start >= 1 && start <= PANELS.length ? start - 1 : 0);
})();
