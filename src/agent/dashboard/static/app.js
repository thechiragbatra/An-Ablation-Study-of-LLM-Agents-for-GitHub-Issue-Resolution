// Ablation dashboard. Vanilla JS, no build step, no external requests.
// Trace data contains arbitrary code and model output, so it is only ever rendered as
// text (DOM APIs / textContent), never via innerHTML.
"use strict";

const VARIANTS = ["A0", "A1", "A2", "A3"];
const VARIANT_INFO = {
  A0: "Single-shot patch",
  A1: "Retry with test feedback",
  A2: "K candidates + critic",
  A3: "Critic reflection loop",
};
const RUBRIC = [
  ["correctness_likelihood", "correctness"],
  ["minimality", "minimality"],
  ["test_impact", "test impact"],
  ["risk", "risk (higher = worse)"],
];
const TERMINAL = ["sandbox_verified", "unverified", "budget_exceeded", "error"];

const state = {
  runs: [],
  filters: { variant: "", status: "", model: "", validOnly: true, hideScripted: false },
  sort: { key: "ts", dir: -1 },
};

// ------------------------------------------------------------------ DOM helpers

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  for (const c of children.flat(Infinity)) {
    if (c == null) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

const app = () => document.getElementById("app");

// ------------------------------------------------------------------ formatting

const fmtUsd = (x) => (x == null ? "—" : x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`);
const fmtNum = (x) => (x == null ? "—" : x.toFixed(1));
const fmtInt = (x) => (x == null ? "—" : Math.round(x).toLocaleString());
const fmtSec = (x) => (x == null ? "—" : x < 1 ? `${(x * 1000).toFixed(0)} ms` : `${x.toFixed(1)} s`);
const fmtMs = (x) => (x == null ? "—" : fmtSec(x / 1000));
const fmtPct = (k, n) => (n === 0 ? "—" : `${((100 * k) / n).toFixed(0)}%`);
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
const mean = (xs) => {
  const v = xs.filter((x) => typeof x === "number");
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const statusLabel = (st) => (st || "unknown").replaceAll("_", " ");
const badge = (st) => h("span", { class: `badge ${st || ""}` }, statusLabel(st));

// ------------------------------------------------------------------ data access

async function api(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  let body = null;
  try {
    body = await res.json();
  } catch {
    // fall through to the status check
  }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

function errorBox(err) {
  return h("div", { class: "banner bad" }, `Could not load data: ${err.message}`);
}

// ------------------------------------------------------------------ routing

function route() {
  const m = location.hash.match(/^#\/run\/(.+)$/);
  if (m) renderRun(decodeURIComponent(m[1]));
  else renderOverview();
}

// ------------------------------------------------------------------ overview

async function renderOverview() {
  app().replaceChildren(h("p", { class: "muted" }, "Loading runs…"));
  try {
    state.runs = await api("/api/runs");
  } catch (err) {
    app().replaceChildren(errorBox(err));
    return;
  }
  drawOverview();
}

function filtered() {
  const f = state.filters;
  return state.runs.filter(
    (r) =>
      (!f.variant || r.variant === f.variant) &&
      (!f.status || r.status === f.status) &&
      (!f.model || r.model === f.model) &&
      (!f.validOnly || r.trace_valid) &&
      (!f.hideScripted || !r.scripted),
  );
}

function drawOverview() {
  const runs = filtered();
  const stats = runs.filter((r) => r.trace_valid && TERMINAL.includes(r.status));
  const nScripted = runs.filter((r) => r.scripted).length;
  app().replaceChildren(
    filterBar(),
    state.runs.length === 0
      ? h(
          "div",
          { class: "panel" },
          h("h2", {}, "No traces yet"),
          h("p", { class: "muted" }, "Run an instance, e.g. "),
          h("pre", {}, "uv run python -m agent.run --config configs/a1.yaml \\\n    --instance tests/fixtures/instances/calc_001.json"),
          h("p", { class: "muted" }, "or generate scripted demo traces with ", h("code", {}, "uv run python scripts/demo_traces.py"), "."),
        )
      : null,
    nScripted
      ? h("div", { class: "banner bad" }, `${nScripted} of ${runs.length} shown runs are scripted test runs (FakeLLM). They exercise the pipeline but are not experimental results.`)
      : null,
    h("section", { class: "cards" }, VARIANTS.map((v) => variantCard(v, stats.filter((r) => r.variant === v)))),
    h(
      "section",
      { class: "panel" },
      h("h2", {}, "Cost vs sandbox-verified rate (per variant)"),
      scatter(stats),
      h("p", { class: "muted" }, "Valid, finished traces only. Bubble size is the number of runs. Formal resolve rates, CIs and McNemar tests come with the Phase 3 harness."),
    ),
    h("section", { class: "panel" }, h("h2", {}, `Runs (${runs.length})`), runsTable(runs)),
  );
}

function uniq(key) {
  return [...new Set(state.runs.map((r) => r[key]).filter((x) => x != null))].sort();
}

function filterBar() {
  const f = state.filters;
  const select = (label, key, values) =>
    h(
      "label",
      {},
      label,
      h(
        "select",
        { onchange: (e) => { f[key] = e.target.value; drawOverview(); } },
        h("option", { value: "" }, "all"),
        values.map((v) => h("option", { value: v, selected: f[key] === v }, statusLabel(v))),
      ),
    );
  const toggle = (label, key) =>
    h(
      "label",
      {},
      h("input", { type: "checkbox", checked: f[key], onchange: (e) => { f[key] = e.target.checked; drawOverview(); } }),
      label,
    );
  return h(
    "div",
    { class: "panel filters" },
    select("Variant", "variant", uniq("variant")),
    select("Status", "status", uniq("status")),
    select("Model", "model", uniq("model")),
    toggle("valid traces only", "validOnly"),
    toggle("hide scripted runs", "hideScripted"),
  );
}

function variantCard(v, runs) {
  const n = runs.length;
  const k = runs.filter((r) => r.status === "sandbox_verified").length;
  return h(
    "div",
    { class: `card v-${v}` },
    h("div", {}, h("b", {}, v), " ", h("span", { class: "muted" }, VARIANT_INFO[v])),
    h("div", { class: "big" }, fmtPct(k, n)),
    h("div", { class: "muted" }, n ? `${k}/${n} sandbox-verified` : "no runs"),
    h(
      "dl",
      {},
      h("dt", {}, "mean cost"), h("dd", {}, fmtUsd(mean(runs.map((r) => r.cost_usd)))),
      h("dt", {}, "mean wall time"), h("dd", {}, fmtSec(mean(runs.map((r) => r.wall_time_s)))),
      h("dt", {}, "mean LLM calls"), h("dd", {}, fmtNum(mean(runs.map((r) => r.llm_calls)))),
      h("dt", {}, "budget aborts"), h("dd", {}, String(runs.filter((r) => r.status === "budget_exceeded").length)),
    ),
  );
}

function scatter(runs) {
  const pts = VARIANTS.map((v) => {
    const rs = runs.filter((r) => r.variant === v);
    if (!rs.length) return null;
    const k = rs.filter((r) => r.status === "sandbox_verified").length;
    return { v, n: rs.length, x: mean(rs.map((r) => r.cost_usd)) || 0, y: k / rs.length };
  }).filter(Boolean);
  if (!pts.length) return h("p", { class: "muted" }, "No finished runs match the filters.");

  const W = 560, H = 260, L = 48, R = 20, T = 14, B = 36;
  const xMax = Math.max(...pts.map((p) => p.x)) * 1.15 || 1;
  const nMax = Math.max(...pts.map((p) => p.n));
  const sx = (x) => L + (x / xMax) * (W - L - R);
  const sy = (y) => T + (1 - y) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return s(
    "svg",
    { class: "scatter", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Mean cost versus verified rate per variant" },
    ticks.map((t) => [
      s("line", { class: "axis", x1: L, x2: W - R, y1: sy(t), y2: sy(t) }),
      s("text", { x: L - 6, y: sy(t) + 4, "text-anchor": "end" }, `${t * 100}%`),
    ]),
    [0, 0.5, 1].map((t) => s("text", { x: sx(t * xMax), y: H - B + 16, "text-anchor": "middle" }, fmtUsd(t * xMax))),
    s("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }, "mean cost per run"),
    pts.map((p) => [
      s("circle", { class: `pt v-${p.v}`, cx: sx(p.x), cy: sy(p.y), r: 6 + 10 * Math.sqrt(p.n / nMax) },
        s("title", {}, `${p.v}: ${(p.y * 100).toFixed(0)}% verified, ${fmtUsd(p.x)} mean cost, n=${p.n}`)),
      s("text", { class: "lbl", x: sx(p.x) + 14, y: sy(p.y) - 8 }, `${p.v} (n=${p.n})`),
    ]),
  );
}

const COLUMNS = [
  ["ts", "Started", (r) => fmtTime(r.ts)],
  ["run_id", "Run", (r) => [h("a", { href: `#/run/${encodeURIComponent(r.run_id)}`, class: "mono" }, r.run_id), r.scripted ? h("span", { class: "tag scripted" }, "scripted") : null]],
  ["variant", "Variant", (r) => r.variant || "—"],
  ["instance_id", "Instance", (r) => h("span", { class: "mono" }, r.instance_id || "—")],
  ["status", "Status", (r) => badge(r.status)],
  ["attempts", "Rounds", (r) => fmtInt(r.attempts), "num"],
  ["cost_usd", "Cost", (r) => fmtUsd(r.cost_usd), "num"],
  ["tokens_in", "Tokens in/out", (r) => `${fmtInt(r.tokens_in)} / ${fmtInt(r.tokens_out)}`, "num"],
  ["llm_calls", "LLM calls", (r) => fmtInt(r.llm_calls), "num"],
  ["sandbox_runs", "Sandbox runs", (r) => fmtInt(r.sandbox_runs), "num"],
  ["wall_time_s", "Wall", (r) => fmtSec(r.wall_time_s), "num"],
  ["trace_valid", "Trace", (r) => (r.trace_valid ? h("span", { class: "badge ok" }, "valid") : h("span", { class: "badge bad", title: r.problems.join("\n") }, "⚠ invalid"))],
];

function runsTable(runs) {
  const { key, dir } = state.sort;
  const sorted = [...runs].sort((a, b) => {
    const x = a[key], y = b[key];
    if (x == null) return 1;
    if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });
  const head = h(
    "tr",
    {},
    COLUMNS.map(([k, label, , cls]) =>
      h(
        "th",
        {
          class: `sortable ${cls || ""}`,
          "aria-sort": k === key ? (dir > 0 ? "ascending" : "descending") : null,
          onclick: () => {
            state.sort = { key: k, dir: k === key ? -dir : -1 };
            drawOverview();
          },
        },
        label,
        k === key ? (dir > 0 ? " ▲" : " ▼") : "",
      ),
    ),
  );
  const rows = sorted.map((r) =>
    h(
      "tr",
      { class: "clickable", onclick: (e) => { if (e.target.tagName !== "A") location.hash = `#/run/${encodeURIComponent(r.run_id)}`; } },
      COLUMNS.map(([, , render, cls]) => h("td", { class: cls || null }, render(r))),
    ),
  );
  return h("table", {}, h("thead", {}, head), h("tbody", {}, rows));
}

// ------------------------------------------------------------------ run detail

async function renderRun(runId) {
  app().replaceChildren(h("p", { class: "muted" }, "Loading run…"));
  let data;
  try {
    data = await api(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (err) {
    app().replaceChildren(h("a", { href: "#/" }, "← All runs"), errorBox(err));
    return;
  }
  const { summary: sm, records, final_state: fs } = data;
  const nodes = records.filter((r) => r.event === "node");
  const reflections = fs ? fs.critic_notes.filter((n) => n.kind === "reflection") : [];
  app().replaceChildren(
    h("a", { href: "#/" }, "← All runs"),
    runHeader(sm),
    sm.problems.length
      ? h("div", { class: "banner bad" }, h("b", {}, "Trace invalid: excluded from analysis."), h("ul", {}, sm.problems.map((p) => h("li", {}, p))))
      : h("div", { class: "banner ok" }, "Trace valid: complete, contiguous, and totals reconcile."),
    sm.error ? h("div", { class: "banner bad" }, h("b", {}, "Run error: "), h("span", { class: "mono" }, sm.error)) : null,
    h("section", { class: "panel" }, h("h2", {}, `Timeline (${nodes.length} steps)`), timeline(nodes)),
    h(
      "section",
      { class: "panel" },
      h("h2", {}, "Candidates"),
      fs ? candidatesView(fs) : h("p", { class: "muted" }, "No final state was recorded for this run (older trace)."),
    ),
    reflections.length
      ? h(
          "section",
          { class: "panel" },
          h("h2", {}, "Critic reflections"),
          reflections.map((n) =>
            h("div", {}, h("h3", {}, `After round ${n.attempt}, on candidate #${n.candidate_index}`), h("div", { class: "critique" }, n.critique)),
          ),
        )
      : null,
  );
}

function chip(label, value) {
  return h("span", { class: "chip" }, h("b", {}, label), value == null ? "—" : String(value));
}

function stat(label, value) {
  return h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value));
}

function runHeader(sm) {
  const cap = sm.budget_per_run_usd;
  const used = cap && sm.cost_usd != null ? sm.cost_usd / cap : null;
  return h(
    "section",
    { class: "panel" },
    h("h1", {}, sm.run_id),
    h(
      "div",
      { class: "chips" },
      chip("variant", sm.variant),
      chip("instance", sm.instance_id),
      chip("model", sm.model),
      sm.scripted ? h("span", { class: "tag scripted" }, "scripted (FakeLLM)") : null,
      chip("effort", sm.effort),
      chip("K", sm.K),
      chip("D", sm.D),
      chip("N", sm.N),
      chip("critic", sm.critic_enabled ? "on" : "off"),
      chip("critic sandbox", sm.critic_uses_sandbox ? "on" : "off"),
      chip("seed", sm.seed),
      chip("config", sm.config_hash ? sm.config_hash.slice(0, 12) : null),
      chip("git", sm.git_sha ? sm.git_sha.slice(0, 12) : null),
      sm.git_dirty ? h("span", { class: "tag dirty", title: "Uncommitted changes: not reproducible from git alone" }, "dirty tree") : null,
    ),
    h(
      "div",
      { class: "stats" },
      stat("status", badge(sm.status)),
      stat("architect rounds", fmtInt(sm.attempts)),
      stat("LLM calls", fmtInt(sm.llm_calls)),
      stat("tokens in / out", `${fmtInt(sm.tokens_in)} / ${fmtInt(sm.tokens_out)}`),
      stat("sandbox runs", fmtInt(sm.sandbox_runs)),
      stat("wall time", fmtSec(sm.wall_time_s)),
      stat(
        `cost (cap ${fmtUsd(cap)})`,
        h("div", {}, fmtUsd(sm.cost_usd), used == null ? null : h("div", { class: `bar ${used > 1 ? "over" : ""}` }, h("span", { style: { width: `${Math.min(100, used * 100)}%` } }))),
      ),
    ),
  );
}

function timeline(nodes) {
  if (!nodes.length) return h("p", { class: "muted" }, "No node records.");
  const maxLat = Math.max(...nodes.map((n) => n.latency_ms), 1);
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, ["#", "Node", "Status after", "Latency", "", "Cost", "Tokens in/out", "LLM calls", "Sandbox execs"].map((c) => h("th", {}, c)))),
    h(
      "tbody",
      {},
      nodes.map((n) => [
        h(
          "tr",
          { class: n.node.startsWith("_") ? "pseudo" : null },
          h("td", { class: "num" }, String(n.step)),
          h("td", {}, h("b", {}, n.node), n.attempt_count ? h("span", { class: "tag" }, `round ${n.attempt_count}`) : null),
          h("td", {}, badge(n.status_after)),
          h("td", {}, h("div", { class: "bar" }, h("span", { style: { width: `${(100 * n.latency_ms) / maxLat}%` } }))),
          h("td", { class: "num" }, fmtMs(n.latency_ms)),
          h("td", { class: "num" }, n.llm_calls.length ? fmtUsd(n.cost_usd) : "—"),
          h("td", { class: "num" }, n.llm_calls.length ? `${fmtInt(n.tokens_in)} / ${fmtInt(n.tokens_out)}` : "—"),
          h(
            "td",
            {},
            n.llm_calls.map((c, i) =>
              c.artifact
                ? h("button", { type: "button", class: "linkish", onclick: () => openArtifact(c.artifact) }, `view call ${i + 1}`, " ")
                : h("span", { class: "muted" }, `call ${i + 1} `),
            ),
          ),
          h(
            "td",
            {},
            n.sandbox_execs.map((e) =>
              h(
                "span",
                { class: `exec ${e.timed_out ? "timeout" : e.exit_code === 0 ? "" : "fail"}`, title: `${e.cmd}\n${fmtMs(e.duration_ms)}` },
                e.timed_out ? "timeout" : `exit ${e.exit_code ?? "?"}`,
              ),
            ),
          ),
        ),
        n.error ? h("tr", {}, h("td", {}), h("td", { colspan: "8", class: "err" }, n.error)) : null,
      ]),
    ),
  );
}

function candidatesView(fs) {
  const cands = fs.candidates || [];
  if (!cands.length) return h("p", { class: "muted" }, "No candidates were generated.");
  const notes = (fs.critic_notes || []).filter((n) => n.kind === "selection");
  const vrs = fs.validation_results || [];
  const rounds = [...new Set(cands.map((c) => c.attempt))].sort((a, b) => a - b);
  return rounds.map((rnd) => {
    const inRound = cands.filter((c) => c.attempt === rnd);
    const depth = inRound[0].reflection_depth;
    return h(
      "div",
      {},
      h("h3", {}, `Round ${rnd}`, depth ? ` · reflection depth ${depth}` : "", ` · ${inRound.length} candidate${inRound.length === 1 ? "" : "s"}`),
      inRound.map((c) =>
        candidateCard(
          c,
          fs.selected_index,
          notes.find((n) => n.candidate_index === c.index && n.attempt === rnd),
          vrs.filter((v) => v.candidate_index === c.index),
        ),
      ),
    );
  });
}

function candidateCard(c, selectedIndex, note, vrs) {
  const final = vrs.find((v) => v.stage === "final");
  const selected = c.selected_by != null;
  return h(
    "div",
    { class: `cand ${selected ? "selected" : ""}` },
    h(
      "div",
      { class: "cand-head" },
      h("span", { class: "idx" }, `#${c.index}`),
      selected ? h("span", { class: "badge ok" }, `★ selected (${c.selected_by})`) : null,
      c.index === selectedIndex ? h("span", { class: "tag" }, "final selection") : null,
      c.apply_error ? h("span", { class: "badge bad" }, "not applicable") : null,
      c.duplicate_of != null ? h("span", { class: "badge warn" }, `duplicate of #${c.duplicate_of}`) : null,
      final ? h("span", { class: `badge ${final.passed ? "ok" : "bad"}` }, final.passed ? "final tests passed" : "final tests failed") : null,
    ),
    h("p", {}, c.rationale),
    c.apply_error ? h("div", { class: "err" }, c.apply_error) : null,
    note ? rubricView(note) : null,
    vrs.map(validationView),
    c.diff
      ? h("details", { open: selected }, h("summary", {}, "Diff"), diffView(c.diff))
      : h(
          "details",
          {},
          h("summary", {}, `Proposed edits (${c.edits.length})`),
          c.edits.map((e) => h("div", {}, h("div", { class: "mono" }, e.path), h("pre", { class: "diff" }, `search:\n${e.search}\nreplace:\n${e.replace}`))),
        ),
  );
}

function rubricView(note) {
  const sc = note.scores || {};
  return h(
    "div",
    {},
    Object.keys(sc).length
      ? h(
          "div",
          { class: "rubric" },
          RUBRIC.map(([key, label]) => [
            h("span", { class: "muted" }, label),
            h("div", { class: `bar ${key === "risk" ? "risk" : ""}` }, h("span", { style: { width: `${((sc[key] || 0) / 5) * 100}%` } })),
            h("span", { class: "mono" }, sc[key] == null ? "—" : String(sc[key])),
          ]),
        )
      : null,
    note.critique ? h("div", { class: "critique" }, note.critique) : null,
  );
}

function validationView(v) {
  return h(
    "details",
    {},
    h(
      "summary",
      {},
      h("span", { class: `badge ${v.passed ? "ok" : "bad"}` }, `${v.stage} · ${v.passed ? "passed" : "failed"}`),
      " ",
      v.applied ? `exit ${v.exit_code ?? "?"}` : "patch not applied",
      v.timed_out ? " · timed out" : "",
      ` · ${fmtSec(v.duration_s)}`,
    ),
    h("div", { class: "mono muted" }, `$ ${v.command}`),
    v.stdout_tail ? h("pre", { class: "diff" }, v.stdout_tail) : null,
    v.stderr_tail ? h("pre", { class: "diff" }, v.stderr_tail) : null,
  );
}

function diffView(diff) {
  const pre = h("pre", { class: "diff" });
  for (const line of diff.replace(/\n$/, "").split("\n")) {
    let cls = "";
    if (/^(diff --git|index |\+\+\+ |--- |new file mode|deleted file mode)/.test(line)) cls = "meta";
    else if (line.startsWith("@@")) cls = "hunk";
    else if (line.startsWith("+")) cls = "add";
    else if (line.startsWith("-")) cls = "del";
    pre.append(h("span", { class: `dl ${cls}` }, line || " "));
  }
  return pre;
}

// ------------------------------------------------------------------ artifact modal

async function openArtifact(path) {
  const dialog = document.getElementById("artifact-modal");
  const body = document.getElementById("modal-body");
  body.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  if (!dialog.open) dialog.showModal();
  let a;
  try {
    a = await api(`/api/artifact?path=${encodeURIComponent(path)}`);
  } catch (err) {
    body.replaceChildren(errorBox(err));
    return;
  }
  document.getElementById("modal-title").textContent = `LLM call · ${a.purpose || "?"}`;
  let response = a.response_text || "";
  try {
    response = JSON.stringify(JSON.parse(response), null, 2);
  } catch {
    // not JSON; show as-is
  }
  const u = a.usage || {};
  body.replaceChildren(
    h(
      "div",
      { class: "chips" },
      chip("model", a.model),
      chip("effort", a.effort),
      chip("max_tokens", a.max_tokens),
      chip("stop", a.stop_reason),
      chip("in", u.input_tokens),
      chip("out", u.output_tokens),
      chip("cache read", u.cache_read_tokens),
      chip("cost", fmtUsd(a.cost_usd)),
    ),
    h("div", { class: "mono muted" }, path),
    h("section", {}, h("h3", {}, "System"), h("pre", {}, a.system || "")),
    (a.messages || []).map((m) => h("section", {}, h("h3", {}, m.role), h("pre", {}, m.content))),
    h("section", {}, h("h3", {}, "Response"), h("pre", {}, response)),
  );
}

// ------------------------------------------------------------------ boot

document.addEventListener("DOMContentLoaded", () => {
  const dialog = document.getElementById("artifact-modal");
  document.getElementById("modal-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close(); // click on backdrop
  });
  document.getElementById("refresh").addEventListener("click", route);
  window.addEventListener("hashchange", route);
  route();
});
