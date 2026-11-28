// nanostat browser demo: the same ideas as edu/nanostat.py, in plain JavaScript (no libraries).
// 1) statistics from scratch  2) a tiny world with answer keys  3) rewards  4) a tiny RL agent.
"use strict";

// ---------- random numbers (seeded) ----------
function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = (m = 0, s = 1) => { const x = Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()); return m + s * x; };
  return { u, normal, int: (lo, hi) => lo + Math.floor(u() * (hi - lo)), pick: (arr) => arr[Math.floor(u() * arr.length)] };
}

// ---------- 1. statistics from scratch ----------
const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
const vari = (x) => { const m = mean(x); return x.reduce((s, v) => s + (v - m) ** 2, 0) / (x.length - 1); };
function erf(x) { // Abramowitz-Stegun 7.1.26 (max error 1.5e-7)
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const ncdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function lgamma(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const ci of c) s += ci / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
function betacf(a, b, x) { let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d); let h = d;
  for (let m = 1; m <= 200; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d); c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; d = 1 / (Math.abs(d) < 1e-30 ? 1e-30 : d);
    c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break; }
  return h; }
function ibeta(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const tcdf = (t, df) => { const x = df / (df + t * t); const p = 0.5 * ibeta(df / 2, 0.5, x); return t > 0 ? 1 - p : p; };
const pT = (t, df, alt) => alt === "greater" ? 1 - tcdf(t, df) : alt === "less" ? tcdf(t, df) : 2 * (1 - tcdf(Math.abs(t), df));
const pZ = (z, alt) => alt === "greater" ? 1 - ncdf(z) : alt === "less" ? ncdf(z) : 2 * (1 - ncdf(Math.abs(z)));
function ranks(x) { const idx = x.map((v, i) => i).sort((i, j) => x[i] - x[j]); const r = new Array(x.length);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && x[idx[j + 1]] === x[idx[i]]) j++;
    for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2 + 1; i = j + 1; } return r; }
const T = {
  student_t: (a, b, alt) => { const na = a.length, nb = b.length, sp = ((na - 1) * vari(a) + (nb - 1) * vari(b)) / (na + nb - 2);
    return pT((mean(a) - mean(b)) / Math.sqrt(sp * (1 / na + 1 / nb)), na + nb - 2, alt); },
  welch_t: (a, b, alt) => { const va = vari(a) / a.length, vb = vari(b) / b.length;
    const df = (va + vb) ** 2 / (va ** 2 / (a.length - 1) + vb ** 2 / (b.length - 1)); return pT((mean(a) - mean(b)) / Math.sqrt(va + vb), df, alt); },
  mann_whitney: (a, b, alt) => { const n1 = a.length, n2 = b.length, r = ranks(a.concat(b)); let u = 0; for (let i = 0; i < n1; i++) u += r[i];
    u -= n1 * (n1 + 1) / 2; const n = n1 + n2, sd = Math.sqrt(n1 * n2 * (n + 1) / 12), d = u - n1 * n2 / 2; return pZ((d - 0.5 * Math.sign(d)) / sd, alt); },
  paired_t: (d, alt) => pT(mean(d) / Math.sqrt(vari(d) / d.length), d.length - 1, alt),
  wilcoxon: (d, alt) => { d = d.filter((v) => v !== 0); const n = d.length, r = ranks(d.map(Math.abs)); let w = -n * (n + 1) / 4;
    d.forEach((v, i) => { if (v > 0) w += r[i]; }); return pZ((w - 0.5 * Math.sign(w)) / Math.sqrt(n * (n + 1) * (2 * n + 1) / 24), alt); },
  pearson: (x, y, alt) => { const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0;
    x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; syy += (y[i] - my) ** 2; });
    const r = sxy / Math.sqrt(sxx * syy); return pT(r * Math.sqrt((x.length - 2) / (1 - r * r)), x.length - 2, alt); },
  spearman: (x, y, alt) => T.pearson(ranks(x), ranks(y), alt),
};
const DESIGN_OF = { student_t: "two_group", welch_t: "two_group", mann_whitney: "two_group", paired_t: "paired", wilcoxon: "paired", pearson: "correlation", spearman: "correlation" };
const METHODS = Object.keys(DESIGN_OF);

// ---------- 2. a tiny world with answer keys (same rules as edu/nanostat.py) ----------
const IGN = { student_t: "ignores pairing", welch_t: "ignores pairing", mann_whitney: "ignores pairing" };
const RULES = {
  "two_group/none": [["welch_t", "student_t", "mann_whitney"], {}],
  "two_group/unequal_var": [["welch_t"], { student_t: "pooled variance, unequal variances" }],
  "two_group/outliers": [["mann_whitney"], { student_t: "outlier-driven", welch_t: "outlier-driven" }],
  "paired/none": [["paired_t", "wilcoxon"], IGN],
  "paired/outliers": [["wilcoxon"], Object.assign({}, IGN, { paired_t: "outlier-driven" })],
  "correlation/none": [["pearson", "spearman"], {}],
  "correlation/outliers": [["spearman"], { pearson: "leverage-driven" }],
};
const WORLDS = Object.keys(RULES);
const TEXTBOOK = { two_group: "student_t", paired: "paired_t", correlation: "pearson" };
const EZ = { null: 0, small: 2.5, large: 4 };

function contaminate(x, R, sign) { x = x.slice(); const sd = Math.sqrt(vari(x)), med = x.slice().sort((a, b) => a - b)[x.length >> 1];
  for (let k = 0; k < 2; k++) { const i = R.int(0, x.length), s = sign == null ? R.pick([-1, 1]) : sign; x[i] = med + s * (8 + 4 * R.u()) * sd; } return x; }

function makeTask(R, world, effect, alt) {
  world = world || R.pick(WORLDS); effect = effect || R.pick(["null", "small", "large"]); alt = alt || R.pick(["two-sided", "two-sided", "greater", "less"]);
  const [kind, pert] = world.split("/"), z = EZ[effect], sgn = alt === "greater" ? 1 : alt === "less" ? -1 : R.pick([-1, 1]), n = R.int(15, 41);
  const arr = (k, m, s) => Array.from({ length: k }, () => R.normal(m, s)); let data;
  if (kind === "two_group") { const uv = pert === "unequal_var", na = uv ? 2 * n : n, nb = n, sb = uv ? 3 : 1, delta = sgn * z * Math.sqrt(1 / na + sb * sb / nb);
    let a = arr(na, 0, 1), b = arr(nb, delta, sb);
    if (pert === "outliers") { b = contaminate(b, R, effect === "null" ? null : sgn); if (effect === "null") a = contaminate(a, R, null); }
    data = { a: b, b: a };
  } else if (kind === "paired") { let d = arr(n, sgn * z / Math.sqrt(n), 1); if (pert === "outliers") d = contaminate(d, R, effect === "null" ? null : sgn);
    const pre = arr(n, 50, 5); data = { pre, post: pre.map((v, i) => v + d[i]) };
  } else { const r = sgn * Math.tanh(z / Math.sqrt(n - 3)), x = arr(n, 0, 1), y = x.map((v) => r * v + Math.sqrt(1 - r * r) * R.normal());
    if (pert === "outliers") for (let k = 0; k < 2; k++) { x[k] = 8 + 4 * R.u(); y[k] = (effect === "null" ? R.pick([-1, 1]) : sgn) * (8 + 4 * R.u()); }
    data = { x, y }; }
  const [A, B] = RULES[world];
  return { world, kind, pert, effect, alt, data, A, B };
}

function execute(task, design, method, alt) {
  if (DESIGN_OF[method] !== design) return null;
  const d = task.data, k = task.kind;
  if (design === "two_group" && k !== "correlation") return k === "two_group" ? T[method](d.a, d.b, alt) : T[method](d.post, d.pre, alt);
  if (design === "paired" && k === "paired") return T[method](d.post.map((v, i) => v - d.pre[i]), alt);
  if (design === "correlation" && k === "correlation") return T[method](d.x, d.y, alt);
  return null;
}

// ---------- 3. rewards ----------
function zc(p) { if (p <= 0) return 5; let lo = -40, hi = 40; for (let i = 0; i < 70; i++) { const m = (lo + hi) / 2; if (ncdf(m) < p / 2) lo = m; else hi = m; } return Math.min(5, -(lo + hi) / 2); }
function fisherReward(task, design, method, alt) {
  const p = execute(task, design, method, alt), ref = execute(task, task.kind, task.A[0], task.alt);
  if (p == null) return 0; return 0.9 * Math.exp(-Math.abs(zc(p) - zc(ref))) + 0.1 * ((p < 0.05) === (ref < 0.05));
}
function ctvReward(task, design, method, alt) {
  const p = execute(task, design, method, alt); if (p == null) return -0.5;
  if (method in task.B || alt !== task.alt) return -0.2; if (!task.A.includes(method)) return 0;
  const refs = task.A.map((m) => execute(task, task.kind, m, task.alt));
  return 0.5 + 0.4 * Math.max(...refs.map((r) => Math.exp(-Math.abs(zc(p) - zc(r))))) + 0.1 * ((p < 0.05) === (refs[0] < 0.05));
}

// ---------- 4. a tiny RL agent: a softmax policy over methods for each observed situation ----------
// state = what the agent can see: design + "outliers?" + "unequal spread?" (no answers, no p-values)
function observe(task) { const rz = (v) => { const s = v.slice().sort((a, b) => a - b), m = s[s.length >> 1];
    const mad = v.map((x) => Math.abs(x - m)).sort((a, b) => a - b)[v.length >> 1] || 1; return Math.max(...v.map((x) => Math.abs(0.6745 * (x - m) / mad))); };
  const k = task.kind, d = task.data, vals = k === "two_group" ? [d.a, d.b] : k === "paired" ? [d.post.map((v, i) => v - d.pre[i])] : [d.x, d.y];
  const outl = Math.max(...vals.map(rz)) > 5; let spread = "";
  if (k === "two_group") { const r = vari(d.a) / vari(d.b); spread = (r > 4 || r < 0.25) ? "|var:high" : "|var:low"; }
  return k + (outl ? "|outl:yes" : "|outl:no") + spread; }
const ACTIONS = { two_group: ["student_t", "welch_t", "mann_whitney"], paired: ["paired_t", "wilcoxon", "student_t", "welch_t", "mann_whitney"], correlation: ["pearson", "spearman"] };

class Policy { // logits[state][method]; starts with the "textbook" habit (SFT)
  constructor() { this.L = {}; }
  logits(s) { if (!this.L[s]) { const k = s.split("|")[0]; this.L[s] = Object.fromEntries(ACTIONS[k].map((m) => [m, m === TEXTBOOK[k] ? 3.0 : 0])); } return this.L[s]; }
  probs(s) { const l = this.logits(s), mx = Math.max(...Object.values(l)), e = Object.entries(l).map(([m, v]) => [m, Math.exp(v - mx)]);
    const z = e.reduce((a, [, v]) => a + v, 0); return Object.fromEntries(e.map(([m, v]) => [m, v / z])); }
  sample(s, R) { const p = this.probs(s); let u = R.u(); for (const [m, v] of Object.entries(p)) { u -= v; if (u <= 0) return m; } return Object.keys(p)[0]; }
  greedy(s) { const p = this.probs(s); return Object.keys(p).reduce((a, b) => (p[a] >= p[b] ? a : b)); }
  update(s, m, adv, lr) { const p = this.probs(s), l = this.logits(s); for (const k of Object.keys(l)) l[k] += lr * adv * ((k === m ? 1 : 0) - p[k]); }
}

function grpoStep(pol, tasks, R, reward, group, lr) { // one GRPO update on `tasks.length` tasks x `group` samples
  for (const t of tasks) { const s = observe(t), ms = Array.from({ length: group }, () => pol.sample(s, R));
    const r = ms.map((m) => reward(t, DESIGN_OF[m], m, t.alt)), mu = mean(r), sd = Math.sqrt(r.reduce((a, v) => a + (v - mu) ** 2, 0) / r.length) + 1e-6;
    ms.forEach((m, i) => pol.update(s, m, (r[i] - mu) / sd, lr)); } }

function evaluate(pol, tasks) { let valid = 0, invalid = 0;
  for (const t of tasks) { const m = pol.greedy(observe(t)); if (t.A.includes(m) && DESIGN_OF[m] === t.kind) valid++; if (m in t.B) invalid++; }
  return { validity: valid / tasks.length, invalid: invalid / tasks.length }; }

// ---------- UI helpers ----------
const $ = (id) => document.getElementById(id);
function bars(el, items, line) { // items: [{label, value, cls}] values in [0,1]
  const max = Math.max(0.2, ...items.map((i) => i.value));
  el.innerHTML = items.map((i) => `<div class="bar-row"><span class="bar-label">${i.label}</span><span class="bar-track">
    <span class="bar ${i.cls || ""}" style="width:${(100 * i.value / max).toFixed(1)}%"></span>
    ${line != null ? `<span class="bar-line" style="left:${(100 * line / max).toFixed(1)}%"></span>` : ""}</span>
    <span class="bar-val">${(100 * i.value).toFixed(1)}%</span></div>`).join("");
}

// Panel 1: when do tests lie?
function runNull() {
  const world = $("null-world").value, R = rng(+$("null-seed").value || 1), N = 500, hits = {}, runs = {};
  for (let i = 0; i < N; i++) { const t = makeTask(R, world, "null", "two-sided");
    for (const m of METHODS) { const p = execute(t, DESIGN_OF[m], m, "two-sided"); if (p != null) { runs[m] = (runs[m] || 0) + 1; hits[m] = (hits[m] || 0) + (p < 0.05); } } }
  const [A, B] = RULES[world];
  bars($("null-out"), Object.keys(runs).map((m) => ({ label: m + (A.includes(m) ? " ✓" : m in B ? " ✗" : ""), value: hits[m] / runs[m],
    cls: A.includes(m) ? "ok" : m in B ? "bad" : "" })), 0.05);
  $("null-note").textContent = `500 datasets with NO real effect. A trustworthy test rejects about 5% of the time (the line). ✓ = admissible here, ✗ = invalid here.`;
}

// Panel 2: reward playground
function runRewards() {
  const R = rng(+$("rw-seed").value || 7), t = makeTask(R, $("rw-world").value, $("rw-effect").value, null);
  const rows = []; const designs = { two_group: ["student_t", "welch_t", "mann_whitney"], paired: ["paired_t", "wilcoxon"], correlation: ["pearson", "spearman"] };
  for (const [d, ms] of Object.entries(designs)) for (const m of ms) rows.push([d, m, t.alt]);
  rows.push([t.kind, t.A[0], t.alt === "greater" ? "less" : "greater"]);
  $("rw-task").innerHTML = `<b>World:</b> ${t.world} · <b>true effect:</b> ${t.effect} · <b>claim:</b> ${t.alt} · <b>admissible:</b> ${t.A.join(", ")}` +
    (Object.keys(t.B).length ? ` · <b>invalid:</b> ${Object.entries(t.B).map(([m, r]) => `${m} (${r})`).join(", ")}` : "");
  $("rw-out").innerHTML = `<table><tr><th>plan (design · method · tail)</th><th>p-value</th><th>Fisher-style reward</th><th>CTV reward</th></tr>` +
    rows.map(([d, m, a]) => { const p = execute(t, d, m, a), f = fisherReward(t, d, m, a), c = ctvReward(t, d, m, a);
      const flag = m in t.B && f > 0.5 ? " class=\"hl\"" : "";
      return `<tr${flag}><td>${d} · ${m} · ${a}</td><td>${p == null ? "cannot run" : p.toFixed(4)}</td><td>${f.toFixed(2)}</td><td>${c.toFixed(2)}</td></tr>`; }).join("") + "</table>";
}

// Panel 3: train a tiny agent with both rewards at the same compute
let training = false;
function runTrain() {
  if (training) return; training = true; $("tr-go").disabled = true;
  const seed = +$("tr-seed").value || 3, steps = +$("tr-steps").value || 150, R0 = rng(seed);
  const train = Array.from({ length: 1500 }, () => makeTask(R0)), RT = rng(seed * 1000 + 1), test = Array.from({ length: 400 }, () => makeTask(RT));
  const arms = { fisher: { pol: new Policy(), R: rng(seed + 11), rw: fisherReward, curve: [] }, ctv: { pol: new Policy(), R: rng(seed + 11), rw: ctvReward, curve: [] } };
  for (const a of Object.values(arms)) a.curve.push(evaluate(a.pol, test).validity);
  let step = 0; const batch = 8, group = 8, lr = 0.03;
  function tick() {
    for (let k = 0; k < 5 && step < steps; k++, step++) for (const a of Object.values(arms)) {
      const picked = Array.from({ length: batch }, () => train[a.R.int(0, train.length)]); grpoStep(a.pol, picked, a.R, a.rw, group, lr);
      a.curve.push(evaluate(a.pol, test).validity); }
    draw(arms, steps); $("tr-step").textContent = `step ${step}/${steps} · each arm generated ${step * batch * group} plans`;
    if (step < steps) requestAnimationFrame(tick); else { training = false; $("tr-go").disabled = false;
      const e = Object.fromEntries(Object.entries(arms).map(([k, a]) => [k, evaluate(a.pol, test)]));
      $("tr-final").innerHTML = `Final (greedy, 400 test tasks): <b>Fisher-style</b> validity ${(100 * e.fisher.validity).toFixed(1)}%, invalid ${(100 * e.fisher.invalid).toFixed(1)}% · ` +
        `<b>CTV</b> validity ${(100 * e.ctv.validity).toFixed(1)}%, invalid ${(100 * e.ctv.invalid).toFixed(1)}% · started from ${(100 * arms.ctv.curve[0]).toFixed(1)}% (textbook habit). ` +
        `Watch the curves, not just the end: in this tiny world both can get there eventually; the question is how many plans it costs.`; } }
  tick();
}
function draw(arms, steps) {
  const c = $("tr-canvas"), g = c.getContext("2d"), W = c.width, H = c.height, css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, W, H); g.strokeStyle = css.getPropertyValue("--grid"); g.lineWidth = 1;
  for (let y = 0; y <= 1.0001; y += 0.25) { const py = H - 20 - y * (H - 30); g.beginPath(); g.moveTo(36, py); g.lineTo(W - 8, py); g.stroke();
    g.fillStyle = css.getPropertyValue("--muted"); g.font = "11px system-ui"; g.fillText(`${y * 100}%`, 2, py + 4); }
  const col = { fisher: css.getPropertyValue("--c2"), ctv: css.getPropertyValue("--c1") };
  for (const [k, a] of Object.entries(arms)) { g.strokeStyle = col[k]; g.lineWidth = 2; g.beginPath();
    a.curve.forEach((v, i) => { const px = 36 + (W - 44) * i / steps, py = H - 20 - v * (H - 30); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); }
}

document.addEventListener("DOMContentLoaded", () => {
  const opts = WORLDS.map((w) => `<option>${w}</option>`).join("");
  $("null-world").innerHTML = opts; $("rw-world").innerHTML = opts;
  $("null-world").value = "two_group/unequal_var"; $("rw-world").value = "two_group/outliers";
  $("null-go").onclick = runNull; $("rw-go").onclick = runRewards; $("tr-go").onclick = runTrain;
  runNull(); runRewards();
});
// exported for tests / curious readers in the console
window.nanostat = { T, makeTask, execute, fisherReward, ctvReward, rng };
