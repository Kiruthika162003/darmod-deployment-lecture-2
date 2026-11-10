/* Maple front-end.

Tabs: Online / Streaming / Batch / Compare / Test Lab. Each tab keeps its own
transcript and input. Same backend session id is used everywhere so the
ADK conversation memory works across tabs.

Vector Store: catalog items are clickable. Click adds to cart. Filling
name + email enables Place order, which sends one prompt to Maple in
the currently active tab and lets the agent + HITL + admin queue do
the rest.
*/

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  }[c]));

  // ---- Timing: live stopwatch and cold-start tag -----------------------------
  // Where a request's timing is written: the tab's meter line, or the small
  // meta label on a pane in a side-by-side row.
  function timingSetter(bubble, meterEl) {
    const el = meterEl || bubble.parentElement.querySelector(".compare-meta");
    return (text, cold = false) => {
      if (!el) return;
      el.textContent = text;
      el.classList.toggle("cold", cold);
    };
  }

  // Ticks while a request is in flight so the room can watch the wait.
  // Returns a function that stops the clock.
  function startStopwatch(show) {
    const t0 = performance.now();
    const tick = () => show(`${Math.round(performance.now() - t0)} ms`);
    tick();
    const id = setInterval(tick, 100);
    return () => clearInterval(id);
  }

  function ageLabel(s) { return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`; }

  // The server says whether this was the first request its instance served.
  function startNote(cold, age) {
    if (cold === undefined) return "";
    return cold
      ? ` . COLD START (instance up ${ageLabel(age)})`
      : ` . warm (instance up ${ageLabel(age)})`;
  }

  // ---- Steps: what the agent did to produce a reply ------------------------
  // Maple does not just answer. Often she decides to call a tool, waits for it,
  // then writes the answer: each is a model call or a wait the user sits through.
  function fmtSecs(ms) { return ms < 950 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`; }

  const STEP_LABEL = {
    decide: (s) => `Maple decides to call ${s.name}`,
    tool: (s) => `${s.name} returns`,
    answer: () => "Maple writes the answer",
  };

  function renderSteps(bubble, steps, usage) {
    if (!steps?.length) return;
    const calls = usage?.model_calls || 0;
    const tokens = (usage?.input_tokens || 0) + (usage?.output_tokens || 0) + (usage?.thinking_tokens || 0);
    const parts = [`${steps.length} step${steps.length === 1 ? "" : "s"}`];
    if (calls) parts.push(`${calls} model call${calls === 1 ? "" : "s"}`);
    if (tokens) parts.push(`${tokens.toLocaleString()} tokens`);

    const box = document.createElement("details");
    box.className = "steps";
    box.open = !bubble.closest(".compare-wrap");  // collapsed in the narrow Test Lab panes
    const summary = document.createElement("summary");
    summary.textContent = parts.join(" . ");
    box.appendChild(summary);

    const list = document.createElement("ol");
    steps.forEach((s) => {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = STEP_LABEL[s.kind](s);
      const time = document.createElement("span");
      time.className = "step-ms";
      time.textContent = fmtSecs(s.ms);
      li.append(label, time);
      if (s.detail) {
        const detail = document.createElement("div");
        detail.className = "step-detail";
        detail.textContent = s.detail;
        li.appendChild(detail);
      }
      list.appendChild(li);
    });
    box.appendChild(list);

    if (tokens) {
      const t = document.createElement("div");
      t.className = "step-detail";
      t.textContent =
        `tokens: in ${usage.input_tokens.toLocaleString()} . out ${usage.output_tokens.toLocaleString()}` +
        (usage.thinking_tokens ? ` . thinking ${usage.thinking_tokens.toLocaleString()}` : "");
      box.appendChild(t);
    }
    bubble.appendChild(box);
  }

  // The two models the Compare tab sets against each other; the server says which.
  const models = { chat: "chat model", pro: "pro model" };

  // Whether Maple may search the web before answering.
  function useSearch() { return $("useSearch").checked; }

  // ---- Tab state -----------------------------------------------------------
  // Per-tab: which is busy (so we can disable just one tab's input).
  const busy = { online: false, stream: false, batch: false, compare: false, all: false };

  function getActiveTab() {
    const t = document.querySelector(".tab.active");
    return t ? t.dataset.tab : "online";
  }
  function transcriptFor(tab) { return document.querySelector(`[data-transcript="${tab}"]`); }
  function inputFor(tab)      { return document.querySelector(`[data-input="${tab}"]`); }
  function sendFor(tab)       { return document.querySelector(`[data-send="${tab}"]`); }
  function meterFor(tab)      { return document.querySelector(`[data-meter="${tab}"]`); }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const id = `panel-${tab.dataset.tab}`;
      const el = document.getElementById(id);
      if (el) el.classList.add("active");
      inputFor(tab.dataset.tab)?.focus();
    });
  });

  // ---- Transcript helpers --------------------------------------------------
  function addUser(tab, text) {
    const t = transcriptFor(tab);
    const row = document.createElement("div");
    row.className = "row user";
    row.innerHTML = `<div class="who">you</div><div class="bubble">${escapeHtml(text)}</div>`;
    t.appendChild(row);
    scrollTab(tab);
  }

  function addAssistant(tab, label = "Maple") {
    const t = transcriptFor(tab);
    const row = document.createElement("div");
    row.className = "row maple";
    row.innerHTML = `<div class="who">${escapeHtml(label)}</div><div class="bubble"></div>`;
    t.appendChild(row);
    scrollTab(tab);
    return row.querySelector(".bubble");
  }

  function addCompareRow(tab, panes) {
    const t = transcriptFor(tab);
    const row = document.createElement("div");
    row.className = "row maple";
    const wrap = document.createElement("div");
    wrap.className = "compare-wrap";
    panes.forEach((p) => {
      const pane = document.createElement("div");
      pane.className = "compare-pane";
      pane.innerHTML = `
        <div class="compare-head">
          <span class="compare-dot" style="background:${p.color}"></span>
          <span class="compare-label">${escapeHtml(p.label)}</span>
          <span class="compare-meta"></span>
        </div>
        <div class="bubble"></div>`;
      wrap.appendChild(pane);
    });
    row.appendChild(wrap);
    t.appendChild(row);
    scrollTab(tab);
    return Array.from(wrap.querySelectorAll(".bubble"));
  }

  function setMeta(bubble, text) {
    const m = bubble.parentElement.querySelector(".compare-meta");
    if (m) m.textContent = text;
  }

  function scrollTab(tab) {
    const t = transcriptFor(tab);
    if (t) t.scrollTop = t.scrollHeight;
  }

  function setBusy(tab, b) {
    busy[tab] = b;
    const input = inputFor(tab);
    const send  = sendFor(tab);
    if (input) input.disabled = b;
    if (send) {
      send.disabled = b;
      send.dataset.label ||= send.textContent;
      send.textContent = b ? "..." : send.dataset.label;
    }
  }

  // ---- Reply primitives ----------------------------------------------------
  async function askOnline(prompt, bubble, meterEl) {
    const show = timingSetter(bubble, meterEl);
    const stop = startStopwatch(show);
    const t0 = performance.now();
    let r, data;
    try {
      r = await fetch("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ prompt, use_search: useSearch() }),
      });
      data = await r.json().catch(() => ({}));
    } finally {
      stop();
    }
    const dt = Math.round(performance.now() - t0);
    const label = `${dt} ms`;
    if (!r.ok) {
      bubble.textContent = "Error: " + (data.detail || `request failed (HTTP ${r.status})`);
      bubble.classList.add("bubble-error");
      show(label);
      return;
    }
    bubble.textContent = data.reply || "(empty)";
    show(label + startNote(data.cold_start, data.instance_age_s), !!data.cold_start);
    if (data.used_tools?.length) {
      const t = document.createElement("div");
      t.className = "tool-tag";
      t.textContent = "tools: " + data.used_tools.join(", ");
      bubble.appendChild(t);
    }
    renderSteps(bubble, data.steps, data.usage);
  }

  async function askStreaming(prompt, bubble, meterEl) {
    const show = timingSetter(bubble, meterEl);
    const t0 = performance.now();
    let first = null;
    const stop = startStopwatch((t) =>
      show(first === null ? `${t} . waiting for first token` : `${t} . first token ${first} ms`));
    let r;
    try {
      r = await fetch("/chat/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ prompt, use_search: useSearch() }),
      });
    } catch (err) {
      stop();
      throw err;
    }
    if (!r.ok || !r.body) {
      stop();
      bubble.textContent = "(stream failed)";
      return;
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const toolsSeen = [];
    const steps = [];
    let usage = null;
    try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, nl); buf = buf.slice(nl + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const json = line.slice(5).trim();
          if (!json) continue;
          let ev;
          try { ev = JSON.parse(json); } catch { continue; }
          if (ev.type === "token") {
            if (first === null) first = Math.round(performance.now() - t0);
            // appendChild(Text) is what actually streams visibly in DOM
            bubble.appendChild(document.createTextNode(ev.text || ""));
          } else if (ev.type === "step") {
            steps.push(ev);
          } else if (ev.type === "tool_call" && ev.name) {
            if (!toolsSeen.includes(ev.name)) toolsSeen.push(ev.name);
          } else if (ev.type === "awaiting_approval") {
            bubble.appendChild(document.createTextNode("\n[order queued for admin approval]"));
            refreshMyOrders();  // opens the pending-order dialog right away
          } else if (ev.type === "done") {
            stop();
            usage = ev.usage;
            const total = Math.round(performance.now() - t0);
            const label = `first token ${first ?? "-"} ms . total ${total} ms`;
            show(label + startNote(ev.cold_start, ev.instance_age_s), !!ev.cold_start);
          } else if (ev.type === "error") {
            stop();
            bubble.appendChild(document.createTextNode("\nError: " + (ev.message || "stream failed")));
            bubble.classList.add("bubble-error");
            show("error");
          }
        }
      }
    }
    } finally {
      stop();
    }
    if (toolsSeen.length) {
      const t = document.createElement("div");
      t.className = "tool-tag";
      t.textContent = "tools: " + toolsSeen.join(", ");
      bubble.appendChild(t);
    }
    renderSteps(bubble, steps, usage);
  }

  // Many prompts in one request, many replies back together.
  async function askBatch(prompts, bubble, meterEl) {
    const show = timingSetter(bubble, meterEl);
    const n = prompts.length;
    const plural = n === 1 ? "" : "s";
    const stop = startStopwatch((t) => show(`${t} . ${n} prompt${plural} in one request`));
    let r, data;
    try {
      r = await fetch("/chat/batch", {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ prompts, use_search: useSearch() }),
      });
      data = await r.json().catch(() => ({}));
    } finally {
      stop();
    }
    if (!r.ok) {
      const why = typeof data.detail === "string" ? data.detail : `request failed (HTTP ${r.status})`;
      bubble.textContent = "Error: " + why;
      bubble.classList.add("bubble-error");
      show("error");
      return;
    }
    bubble.textContent = "";
    data.items.forEach((it, i) => {
      const row = document.createElement("div");
      row.className = "batch-item";
      const q = document.createElement("div");
      q.className = "batch-q";
      q.textContent = `${i + 1}. ${it.prompt}`;
      const a = document.createElement("div");
      a.className = "batch-a";
      a.textContent = it.error ? "Error: " + it.error : it.reply;
      const ms = document.createElement("div");
      ms.className = "batch-ms";
      ms.textContent = `${it.latency_ms} ms`;
      row.append(q, a, ms);
      bubble.appendChild(row);
    });
    const avg = Math.round(data.total_latency_ms / Math.max(1, data.items.length));
    show(
      `${n} prompt${plural} . total ${data.total_latency_ms} ms . avg ${avg} ms each` +
        startNote(data.cold_start, data.instance_age_s),
      !!data.cold_start,
    );
  }

  async function askCompare(prompt, flashBubble, proBubble) {
    const r = await fetch("/chat/compare", {
      method: "POST",
      headers: { "content-type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ prompt }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = "Error: " + (data.detail || `request failed (HTTP ${r.status})`);
      [flashBubble, proBubble].forEach((b) => {
        b.textContent = msg;
        b.classList.add("bubble-error");
      });
      return;
    }
    fillCompareBubble(flashBubble, data.flash);
    fillCompareBubble(proBubble, data.pro);
  }

  function fillCompareBubble(bubble, reply) {
    if (!reply) { bubble.textContent = "(no response)"; return; }
    if (reply.reply && reply.reply !== "(no text returned)") {
      bubble.textContent = reply.reply;
    } else if (reply.error) {
      bubble.textContent = "Error: " + reply.error;
      bubble.classList.add("bubble-error");
    } else {
      bubble.textContent = reply.reply || "(empty)";
    }
    const n = (v) => (v ?? 0).toLocaleString();
    const tokens = reply.output_tokens != null
      ? ` . in ${n(reply.input_tokens)} out ${n(reply.output_tokens)} thinking ${n(reply.thinking_tokens)}`
      : "";
    const cost = reply.estimated_cents != null ? ` . ${reply.estimated_cents} c` : "";
    setMeta(bubble, `${reply.latency_ms ?? "-"} ms${tokens}${cost}`);
  }

  // ---- Dispatch ------------------------------------------------------------
  async function send(tab) {
    if (busy[tab]) return;
    const input = inputFor(tab);
    const prompt = input.value.trim();
    if (!prompt) return;
    // Batch: one prompt per line, kept in the box so it can be run again.
    const batchPrompts = tab === "batch"
      ? prompt.split("\n").map((l) => l.trim()).filter(Boolean).slice(0, 25)
      : null;
    addUser(tab, batchPrompts ? `batch of ${batchPrompts.length}: ${batchPrompts.join(" | ")}` : prompt);
    if (!batchPrompts) input.value = "";
    setBusy(tab, true);
    try {
      if (tab === "online") {
        const b = addAssistant(tab, "Maple");
        await askOnline(prompt, b, meterFor(tab));
      } else if (tab === "stream") {
        const b = addAssistant(tab, "Maple");
        await askStreaming(prompt, b, meterFor(tab));
      } else if (tab === "batch") {
        const b = addAssistant(tab, "Maple");
        await askBatch(batchPrompts, b, meterFor(tab));
      } else if (tab === "compare") {
        const [flash, pro] = addCompareRow(tab, [
          { label: models.chat, color: "var(--pink)" },
          { label: models.pro,  color: "var(--blue)" },
        ]);
        await askCompare(prompt, flash, pro);
      } else if (tab === "all") {
        const [online, stream, batch] = addCompareRow(tab, [
          { label: "online",           color: "var(--pink)" },
          { label: "streaming",        color: "var(--cyan)" },
          { label: "batch . 1 prompt", color: "var(--orange)" },
        ]);
        await Promise.all([
          askOnline(prompt, online),
          askStreaming(prompt, stream),
          askBatch([prompt], batch),
        ]);
      }
    } catch (err) {
      addAssistant(tab, "error").textContent = String(err);
    } finally {
      setBusy(tab, false);
      input.focus();
      refreshMyOrders();
    }
  }

  document.querySelectorAll(".send").forEach((btn) => {
    btn.addEventListener("click", () => send(btn.dataset.send));
  });
  document.querySelectorAll(".input").forEach((inp) => {
    inp.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      // In the batch box plain Enter starts a new line; Ctrl+Enter runs it.
      if (inp.tagName === "TEXTAREA" && !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      send(inp.dataset.input);
    });
  });

  // ---- Catalog + cart ------------------------------------------------------
  const catalogEl = $("catalog");
  const cartEl    = $("cart");
  const cartNameEl = $("cartName");
  const cartEmailEl = $("cartEmail");
  const cartSubmitEl = $("cartSubmit");
  const cartMeterEl = $("cartMeter");
  let catalogIndex = {};   // id -> item object
  const cart = new Map();  // id -> quantity

  async function loadCatalog() {
    try {
      const r = await fetch("/catalog", { credentials: "same-origin" });
      const data = await r.json();
      catalogEl.innerHTML = "";
      data.items.forEach((it) => {
        catalogIndex[it.id] = it;
        const card = document.createElement("button");
        card.type = "button";
        card.className = "cat";
        card.dataset.id = it.id;
        card.innerHTML = `
          <div class="cat-name">${escapeHtml(it.name)}</div>
          <div class="cat-meta">${it.price_cad ? "$" + it.price_cad + " CAD" : "free"} . ${escapeHtml(it.category)}</div>
          <div class="cat-desc">${escapeHtml(it.summary || "")}</div>
          <div class="cat-add">+ add to cart</div>`;
        card.addEventListener("click", () => addToCart(it.id));
        catalogEl.appendChild(card);
      });
    } catch (err) {
      catalogEl.textContent = "catalog unavailable: " + err;
    }
  }

  function addToCart(id) {
    cart.set(id, (cart.get(id) || 0) + 1);
    renderCart();
  }
  function removeFromCart(id) {
    cart.delete(id);
    renderCart();
  }

  function renderCart() {
    if (cart.size === 0) {
      cartEl.innerHTML = `<div class="empty">Cart is empty. Click an item above.</div>`;
      updateCartSubmitState();
      return;
    }
    let total = 0;
    const rows = [];
    for (const [id, qty] of cart) {
      const it = catalogIndex[id];
      if (!it) continue;
      const sub = (it.price_cad || 0) * qty;
      total += sub;
      rows.push(`
        <div class="cart-row" data-id="${escapeHtml(id)}">
          <span class="cart-name">${escapeHtml(it.name)}${qty > 1 ? " x" + qty : ""}</span>
          <span class="cart-sub">${sub ? "$" + sub : "free"}</span>
          <button class="cart-rm" data-rm="${escapeHtml(id)}" title="remove">x</button>
        </div>`);
    }
    cartEl.innerHTML = rows.join("") +
      `<div class="cart-total">total: ${total ? "$" + total + " CAD" : "free"}</div>`;
    cartEl.querySelectorAll("[data-rm]").forEach((b) => {
      b.addEventListener("click", () => removeFromCart(b.dataset.rm));
    });
    updateCartSubmitState();
  }

  function updateCartSubmitState() {
    const ok = cart.size > 0 && cartNameEl.value.trim() && cartEmailEl.value.trim();
    cartSubmitEl.disabled = !ok;
  }
  cartNameEl.addEventListener("input", updateCartSubmitState);
  cartEmailEl.addEventListener("input", updateCartSubmitState);

  cartSubmitEl.addEventListener("click", async () => {
    if (cart.size === 0) return;
    const items = [];
    for (const [id, qty] of cart) for (let i = 0; i < qty; i++) items.push(id);
    const itemNames = items.map((i) => catalogIndex[i]?.name || i);
    const name = cartNameEl.value.trim();
    const email = cartEmailEl.value.trim();
    const prompt =
      `Place an order for me. Items: ${itemNames.join(", ")}. ` +
      `Name: ${name}. Email: ${email}. Use submit_order; ` +
      `don't ask me to confirm.`;
    // Reuse whichever tab is active so the user sees the request flow.
    const tab = getActiveTab();
    addUser(tab, `[cart] ${itemNames.join(", ")} for ${name}`);
    setBusy(tab, true);
    cartMeterEl.textContent = "placing order...";
    try {
      const b = addAssistant(tab, "Maple");
      if (tab === "stream" || tab === "all") {
        await askStreaming(prompt, b, meterFor(tab));
      } else {
        await askOnline(prompt, b, meterFor(tab));
      }
      cart.clear();
      renderCart();
      cartMeterEl.textContent = "order queued. Check Your orders below.";
    } catch (err) {
      cartMeterEl.textContent = "order failed: " + err;
    } finally {
      setBusy(tab, false);
      refreshMyOrders();
    }
  });

  // ---- My orders -----------------------------------------------------------
  const myOrdersEl = $("myOrders");
  const knownOrders = new Map();
  const confirmed = new Set();  // orders whose Maple confirmation is already in the chat

  async function refreshMyOrders() {
    try {
      const r = await fetch("/orders/my", { credentials: "same-origin" });
      const data = await r.json();
      myOrdersEl.innerHTML = "";
      if (!data.orders || data.orders.length === 0) {
        myOrdersEl.innerHTML = `<div class="empty">No orders yet. Add items to your cart.</div>`;
        return;
      }
      const head = document.createElement("div");
      head.className = "my-orders-head";
      head.textContent = "Your orders";
      myOrdersEl.appendChild(head);
      data.orders.forEach((o) => {
        const wasStatus = knownOrders.get(o.order_id);
        const firstSight = wasStatus === undefined;
        knownOrders.set(o.order_id, o.status);
        const justChanged = wasStatus && wasStatus !== o.status;
        const card = document.createElement("div");
        card.className = `order ${o.status}` + (justChanged ? " just-changed" : "");
        const items = o.items.map((i) => i.name || i.id).join(", ");
        card.innerHTML = `
          <div class="order-line">
            <span class="status status-${o.status}">${o.status.toUpperCase()}</span>
            <span class="order-name">${escapeHtml(o.name || "(no name)")}</span>
            <span class="order-total">${o.total_cad ? "$" + o.total_cad : "free"}</span>
          </div>
          <div class="order-items">${escapeHtml(items)}</div>`;
        myOrdersEl.appendChild(card);
        if (justChanged && o.status === "approved") {
          flashAdmin(o, "approved", `Approved: ${items} for ${o.name || "you"}. It will ship now.`);
        } else if (justChanged && o.status === "denied") {
          flashAdmin(o, "denied", `Denied: ${items}. ${o.reason || ""}`);
        }
        // Maple's own reply to the decision lands a few seconds after it.
        // An order first seen already confirmed (page reload) is not replayed.
        if (o.confirmation && !confirmed.has(o.order_id)) {
          confirmed.add(o.order_id);
          if (!firstSight) showConfirmation(o.confirmation);
        }
      });
      syncOrderModal(data.orders);
    } catch (err) {
      myOrdersEl.textContent = "could not load your orders: " + err;
    }
  }

  function flashAdmin(order, kind, text) {
    // Drop the admin banner into whichever tab the user is currently viewing.
    const tab = getActiveTab();
    const b = addAssistant(tab, `admin . ${kind}`);
    b.parentElement.classList.add("admin-banner", `admin-${kind}`);
    b.textContent = text;
  }

  // Maple answers the admin's decision in the chat.
  function showConfirmation(text) {
    const tab = getActiveTab();
    addAssistant(tab, "Maple").textContent = text;
    scrollTab(tab);
  }

  // ---- Order dialog --------------------------------------------------------
  // Opens when Maple queues an order for approval and settles once a person
  // decides, so the room sees the pause in the agent's loop.
  const modal = $("orderModal");
  const modalBadge = $("orderModalBadge");
  const modalTitle = $("orderModalTitle");
  const modalBody = $("orderModalBody");
  let modalOrder = null;          // order id the dialog is waiting on
  let modalTimer = null;          // closes the dialog after a decision
  const modalHidden = new Set();  // orders the presenter dismissed

  function setModal(kind, title, body) {
    modalBadge.textContent = kind.toUpperCase();
    modalBadge.className = `modal-badge ${kind}`;
    modalTitle.textContent = title;
    modalBody.textContent = body;
    modal.hidden = false;
  }

  function closeModal() {
    clearTimeout(modalTimer);
    modalOrder = null;
    modal.hidden = true;
  }

  $("orderModalHide").addEventListener("click", () => {
    if (modalOrder) modalHidden.add(modalOrder);
    closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) $("orderModalHide").click();
  });

  function syncOrderModal(orders) {
    const summary = (o) => `${o.items.map((i) => i.name || i.id).join(", ")} for ${o.name || "you"}`;
    const pending = orders.find((o) => o.status === "pending" && !modalHidden.has(o.order_id));
    if (pending) {
      clearTimeout(modalTimer);
      modalOrder = pending.order_id;
      setModal(
        "pending",
        "Waiting for a Vector admin",
        `${summary(pending)}. Maple cannot ship this herself: the order waits in an admin queue until a person approves it.`,
      );
      return;
    }
    if (modalOrder) {
      const o = orders.find((x) => x.order_id === modalOrder);
      if (o && o.status !== "pending") {
        const ok = o.status === "approved";
        setModal(
          o.status,
          ok ? "Approved by a Vector admin" : "Denied by a Vector admin",
          ok ? `${summary(o)} will ship now.` : `${summary(o)}. ${o.reason || ""}`,
        );
        modalOrder = null;
        modalTimer = setTimeout(closeModal, 3500);
      }
    }
  }

  // ---- Boot ----------------------------------------------------------------
  const topMeta = $("topMeta");
  const footRegion = $("footRegion");
  const footModel = $("footModel");

  async function boot() {
    try {
      const me = await fetch("/me", { credentials: "same-origin" }).then((r) => r.json());
      topMeta.textContent = `session ${(me.session_id || "?").slice(0, 8)} . ${me.backend} . ${me.model}`;
      footRegion.textContent = `region: ${me.region}`;
      footModel.textContent = `model: ${me.model}`;
      models.chat = me.model;
      models.pro = me.pro_model || models.pro;
    } catch {
      topMeta.textContent = "offline";
    }
    await loadCatalog();
    renderCart();
    await refreshMyOrders();
    setInterval(refreshMyOrders, 2500);
    // Greeting on the default Online tab
    const g = addAssistant("online", "Maple");
    g.textContent =
      "Hi, I'm Maple. Ask me about Vector Institute or the DARMOD program. " +
      "Each tab keeps its own conversation but shares one session memory.";
  }

  boot();
})();
