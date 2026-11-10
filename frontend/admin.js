/* Maple admin page. Password unlocks the order queue; tabs filter status. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const pwEl = $("pw");
  const unlockBtn = $("unlock");
  const statusEl = $("status");
  const listEl = $("list");

  let password = null;
  let activeStatus = "pending";
  let pollTimer = null;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  async function loadList() {
    if (!password) return;
    try {
      const r = await fetch(
        `/admin/orders?status=${encodeURIComponent(activeStatus)}&token=${encodeURIComponent(password)}`,
        { credentials: "same-origin" },
      );
      if (r.status === 401) {
        statusEl.textContent = "Wrong password.";
        password = null;
        listEl.innerHTML = `<div class="admin-empty">Enter the admin password above.</div>`;
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        return;
      }
      const data = await r.json();
      render(data.orders || []);
    } catch (err) {
      statusEl.textContent = "Could not load: " + err;
    }
  }

  function render(orders) {
    if (orders.length === 0) {
      listEl.innerHTML = `<div class="admin-empty">No ${activeStatus} orders.</div>`;
      statusEl.textContent = `Unlocked. 0 ${activeStatus} order(s).`;
      return;
    }
    statusEl.textContent = `Unlocked. ${orders.length} ${activeStatus} order(s). Auto-refreshing every 3s.`;
    listEl.innerHTML = "";
    orders.forEach((o) => {
      const items = o.items.map((i) => i.name || i.id).join(", ");
      const card = document.createElement("div");
      card.className = "admin-card";
      card.innerHTML = `
        <div class="admin-card-head">
          <span class="status status-${o.status}">${o.status.toUpperCase()}</span>
          <strong>${escapeHtml(o.name || "(no name)")}</strong>
          <span class="order-total">${o.total_cad ? "$" + o.total_cad + " CAD" : "free"}</span>
        </div>
        <div class="admin-card-items">${escapeHtml(items)}</div>
        <div class="admin-card-meta">
          ${escapeHtml(o.email || "(no email)")} . order ${escapeHtml(o.order_id)} . queued ${new Date((o.queued_at || 0) * 1000).toLocaleString()}
        </div>`;
      if (o.status === "pending") {
        const actions = document.createElement("div");
        actions.className = "admin-actions";
        actions.innerHTML = `
          <button class="btn-approve" data-id="${escapeHtml(o.order_id)}" data-action="approve">Approve</button>
          <button class="btn-deny"    data-id="${escapeHtml(o.order_id)}" data-action="deny">Deny</button>`;
        actions.addEventListener("click", onAction);
        card.appendChild(actions);
      } else if (o.resolved_at) {
        const note = document.createElement("div");
        note.className = "admin-card-meta";
        note.textContent = `resolved ${new Date(o.resolved_at * 1000).toLocaleString()}${o.reason ? " . " + o.reason : ""}`;
        card.appendChild(note);
      }
      listEl.appendChild(card);
    });
  }

  async function onAction(ev) {
    const btn = ev.target.closest("button");
    if (!btn) return;
    const id = btn.dataset.id;
    const action = btn.dataset.action;
    btn.disabled = true;
    try {
      const r = await fetch(
        `/admin/orders/${encodeURIComponent(id)}/${action}?token=${encodeURIComponent(password)}`,
        { method: "POST", credentials: "same-origin" },
      );
      if (!r.ok) {
        statusEl.textContent = `${action} failed: HTTP ${r.status}`;
      } else {
        await loadList();
      }
    } catch (err) {
      statusEl.textContent = `${action} failed: ${err}`;
    } finally {
      btn.disabled = false;
    }
  }

  function unlock() {
    const pw = pwEl.value.trim();
    if (!pw) return;
    password = pw;
    statusEl.textContent = "Validating...";
    loadList().then(() => {
      if (password && !pollTimer) {
        pollTimer = setInterval(loadList, 3000);
      }
    });
  }

  unlockBtn.addEventListener("click", unlock);
  pwEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); unlock(); }
  });

  document.querySelectorAll(".admin-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".admin-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      activeStatus = tab.dataset.status;
      loadList();
    });
  });
})();
