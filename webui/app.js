"use strict";
/* Shorts Studio front-end. Plain JavaScript, no build step. All text goes through textContent (never innerHTML). */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const PAGE = 24;                                           // moments shown at a time (a very long video can have dozens)
const state = { cfg: null, yt: null, project: null, picked: new Set(), edits: {}, signals: null, jobs: new Map(), dismissed: new Set(),
                logOpen: new Set(), poll: null, players: {}, hits: [], clipPicked: new Set(), ytTimer: null, ytPoll: null, upTimer: null,
                shown: PAGE, tab: "moments", projects: [], uploads: [], ytTab: "waiting", wizardOpen: false };

/* ---------- helpers ---------- */
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.setAttribute("style", v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", "X-Studio": "1", ...(opts.headers || {}) } });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
const clock = s => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
function parseTime(v) {
  v = String(v).trim().replace(",", "."); if (!v) return NaN;
  const parts = v.split(":").map(Number); if (parts.length > 3 || parts.some(Number.isNaN)) return NaN;
  return parts.reduce((a, p) => a * 60 + p, 0);
}
const fmtBytes = n => n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : n >= 1e6 ? Math.round(n / 1e6) + " MB" : n >= 1e3 ? Math.round(n / 1e3) + " KB" : n + " B";
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
function toast(msg, kind = "") {
  const t = el("div", { class: "toast " + kind, text: msg }); $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 9000 : 4500);
}
async function copyText(text, what = "Copied") {
  try { await navigator.clipboard.writeText(text); }
  catch { const ta = el("textarea", { style: "position:fixed;left:-999px" }); ta.value = text; document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove(); }
  toast(what + " to the clipboard", "good");
}
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const localTime = iso => { const d = new Date(iso); return isNaN(d) ? String(iso) : d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); };
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const ytId = () => encodeURIComponent(state.project.id);

/* ---------- theme ---------- */
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  else { t = "auto"; document.documentElement.removeAttribute("data-theme"); }
  $("#theme").textContent = "Theme: " + t; store.set("theme", t); drawTimeline();
}
$("#theme").addEventListener("click", () => { const o = ["auto", "light", "dark"], c = store.get("theme") || "auto"; applyTheme(o[(o.indexOf(c) + 1) % o.length]); });
applyTheme(new URLSearchParams(location.search).get("theme") || store.get("theme") || "auto");

/* ---------- navigation: #/create  #/projects  #/project/<id>  #/youtube ---------- */
const VIEW_IDS = { create: "view-create", projects: "view-projects", project: "project", youtube: "view-youtube" };
const currentIs = name => !$("#" + VIEW_IDS[name]).hidden;
function showView(name) {
  for (const [k, id] of Object.entries(VIEW_IDS)) $("#" + id).hidden = k !== name;
  $$("#nav a").forEach(a => {
    const on = a.dataset.view === name || (name === "project" && a.dataset.view === "projects");
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);
  if (name === "project") requestAnimationFrame(drawTimeline);
  if (name !== "youtube") { clearInterval(state.upTimer); state.upTimer = null; }
}
const go = path => { if (location.hash === "#/" + path) route(); else location.hash = "#/" + path; };
async function route() {
  const [v = "create", ...rest] = location.hash.replace(/^#\/?/, "").split("/");
  const arg = decodeURIComponent(rest.join("/") || "");
  if (v === "project" && arg) {
    try { if (!state.project || state.project.id !== arg) await openProject(arg); showView("project"); }
    catch { toast("That video was not found.", "bad"); go("projects"); }
  } else if (v === "projects") { showView("projects"); loadLibrary(); }
  else if (v === "youtube") { showView("youtube"); loadYouTubePage(); }
  else { showView("create"); loadLibrary(); }
}
window.addEventListener("hashchange", route);

/* ---------- status: header chips, alerts, the "get started" card ---------- */
async function loadConfig() {
  state.cfg = await api("/api/config"); state.yt = state.cfg.youtube || state.yt;
  $("#capScript").value = state.cfg.caption_script || "roman";
  $("#faceBg").value = state.cfg.facecam_bg || "white";
  renderMemeNote();
  const c = state.cfg;
  $("#captions").disabled = $("#titleCard").disabled = $("#watermark").disabled = !c.captions_supported;
  const mine = state.yt && state.yt.channel && state.yt.channel.title;
  $("#watermark").placeholder = mine ? `${mine} (automatic)` : "e.g. @mychannel";
  renderChips(); renderBanner(); renderSetup(); syncAutopilot();
  if (state.project) renderYtBar();                                       // its "waiting / asked to wait" note depends on this status
}
const aiKeySet = () => !!state.cfg && (state.cfg.llm_provider === "gemini" ? state.cfg.gemini_key_set : state.cfg.openai_key_set);
function renderChips() {
  if (!state.cfg) return;
  const c = state.cfg, y = state.yt || {}, chips = [];
  if (!aiKeySet()) chips.push(el("button", { class: "chip bad small", onclick: () => openSettings("ai"), text: "AI key missing: click to add it" }));
  if (!c.ffmpeg) chips.push(el("span", { class: "chip bad", text: "ffmpeg not found" }));
  else if (!c.captions_supported) chips.push(el("span", { class: "chip warn", title: "This ffmpeg build cannot draw text on video", text: "captions unavailable" }));
  if (y.connected) chips.push(el("a", { class: "chip ok", href: "#/youtube", title: "Uploads go to this YouTube channel", text: "▶ " + ((y.channel && y.channel.title) || "YouTube connected") }));
  else if (y.needs_reconnect) chips.push(el("a", { class: "chip warn", href: "#/youtube", text: "YouTube sign-in expired: reconnect" }));
  else chips.push(el("a", { class: "chip", href: "#/youtube", title: "Let the app upload your shorts to your channel", text: "Connect YouTube" }));
  chips.push(el("span", { class: "chip", title: "Shorts Studio version", text: "v" + c.version }));
  $("#chips").replaceChildren(...chips);
  const nq = $("#navYt"); nq.hidden = !(y.queued > 0); nq.textContent = y.queued > 0 ? String(y.queued) : "";
  nq.title = y.queued > 0 ? plural(y.queued, "clip is", "clips are") + " waiting to upload" : "";
  $("#sVersion").textContent = "Shorts Studio v" + c.version;
}
function alertBox(kind, text, ...buttons) {
  return el("div", { class: "alert " + kind }, el("span", { text }), el("div", { class: "grow" }, buttons.filter(Boolean)));
}
function renderMemeNote() {
  const m = state.cfg && state.cfg.memes; if (!m) return;
  const moods = Object.entries(m.by_mood || {}).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(", ");
  const ready = m.usable ?? m.total, off = (m.off || []).length;
  $("#memeNote").textContent = m.total ? `${plural(ready, "meme", "memes")} ready (${moods})${off ? `, ${off} switched off` : ""}.`
    : "No memes yet: drop clips, sounds or pictures into the folder.";
  $("#memeNote").title = off ? "Not used by themselves: " + m.off.join(", ") : "";
}
function renderBanner() {
  const y = state.yt || {}, out = [];
  if (!serverCanEdit()) out.push(alertBox("warn", "The app was updated: quit it and open it again to get the newest editing."));
  if (state.cfg && !aiKeySet()) out.push(alertBox("info", "Add your free AI key first: it is what finds the best moments.", el("button", { class: "small primary", onclick: () => openSettings("ai"), text: "Add AI key" })));
  if (y.needs_reconnect) out.push(alertBox("bad", "Your YouTube sign-in expired" + (y.queued ? `, so ${plural(y.queued, "clip is", "clips are")} waiting.` : "."),
    el("button", { class: "small primary", onclick: () => go("youtube"), text: "Reconnect" })));
  if (y.connected && y.paused) out.push(alertBox("warn", "⏸ Uploads are paused. Nothing is sent to YouTube until you resume.", el("button", { class: "small primary", onclick: () => setPaused(false), text: "Resume uploads" })));
  if (y.connected && y.testing_mode) out.push(alertBox("warn", `Your Google consent screen is still in Testing: the YouTube sign-in stops working in ${Math.max(0, Math.ceil(y.token_days_left || 0))} day(s). Publish it once and it lasts.`,
    el("button", { class: "small", onclick: () => { state.wizardOpen = true; go("youtube"); }, text: "Show me how" })));
  if (y.connected && !y.paused && y.parked_until) out.push(alertBox("info", `${y.parked_reason || "YouTube asked us to wait."} Next try ${localTime(y.parked_until * 1000)}.`,
    el("button", { class: "small primary", onclick: tryUploadsNow, text: "Try now" })));
  $("#banner").replaceChildren(...out);
}
/* stop waiting after YouTube said "not now" and send the queue again (it waits again if YouTube still says no) */
async function tryUploadsNow() {
  try {
    const r = await post("/api/youtube/resume");
    for (const job of r.jobs || []) trackJob(job);
    toast((r.jobs || []).length ? "Trying to upload now…" : "Nothing is waiting to be uploaded.", "");
    await loadConfig(); if (location.hash.startsWith("#/youtube")) loadUploads(); if (state.project) refreshProject();
  } catch (e) { toast(e.message, "bad"); }
}
/* why a queued clip has not gone yet */
function waitingWhy() {
  const y = state.yt || {};
  if (!y.connected) return "waiting: connect YouTube first";
  if (y.paused) return "waiting: uploads are paused";
  if (y.parked_until) return `waiting: YouTube said no for now, next try ${localTime(y.parked_until * 1000)}`;
  if (y.uploads_left === 0) return "waiting: your own limit for today is reached";
  return y.auto_resume ? "waiting: goes within a minute" : "waiting: automatic continuing is off";
}
function renderSetup() {
  const y = state.yt || {}, box = $("#setup"), skip = store.get("skipYt") === "1", hasVideos = state.projects.length > 0;
  const needKey = !aiKeySet(), needYt = !y.connected && !skip;
  box.hidden = !(needKey || needYt);
  if (box.hidden) return;
  const row = (ok, text, ...btns) => el("li", { class: ok ? "ok" : "" }, el("span", { class: "tick", text: ok ? "✓" : "" }), el("span", { text }), el("div", { class: "grow" }, btns.filter(Boolean)));
  box.replaceChildren(el("h2", { text: "Get started" }), el("ol", {},
    row(!needKey, "Add your free AI key (finds the best moments)", needKey ? el("button", { class: "small primary", onclick: () => openSettings("ai"), text: "Add key" }) : null),
    row(!!y.connected, "Connect your YouTube channel (optional: lets the app upload for you)",
      y.connected ? null : el("button", { class: "small primary", onclick: () => go("youtube"), text: "Set up" }),
      y.connected ? null : el("button", { class: "small", onclick: () => { store.set("skipYt", "1"); renderSetup(); }, text: "Skip for now" })),
    row(hasVideos, "Paste a video link below and press the button")));
}

/* ---------- create ---------- */
$("#focusChips").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#focusChips button").forEach(x => x.classList.toggle("on", x === b)); $("#focus").value = b.dataset.focus;
});
$("#focus").addEventListener("input", () => $$("#focusChips button").forEach(x => x.classList.toggle("on", !$("#focus").value && x.dataset.focus === "")));
$("#lenSeg").addEventListener("click", e => { const b = e.target.closest("button"); if (b) $$("#lenSeg button").forEach(x => x.classList.toggle("on", x === b)); });
$("#urls").addEventListener("input", e => { const t = e.target; t.rows = Math.min(6, Math.max(1, t.value.split("\n").length)); });
$("#urls").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#analyze").click(); });

/* the choices you made last time are still there next time */
const FX = ["jumpcut", "zoom", "memes", "popups", "sfx", "hook", "cta", "progress", "color"];
const EDIT_PRESETS = { off: [], clean: ["jumpcut", "zoom", "hook", "progress"], viral: FX.filter(k => k !== "popups" && k !== "cta") };
const EDIT_NOTES = {
  viral: "Pauses cut, punch-in zooms, memes from your folder at the funny bits, sound effects, a headline, animated captions, a progress bar and your channel’s name.",
  clean: "Pauses cut, gentle zooms, animated captions, a headline and a progress bar. No pop-ups or sounds.",
  off: "No editing: the clip as it is, framed for a vertical screen." };
const REMEMBER = ["cands", "lang", "useAudio", "useReplay", "preset", "layout", "quality", "captions", "capStyle", "titleCard", "normalize", "kit", "watermark",
                  "editStyle", ...FX.map(k => "fx_" + k), "apCount", "apMin", "apPrivacy", "apWhen"];       // (Autopilot itself is never remembered: it starts off every time)
/* choosing an editing style sets its effects (each can still be changed under "More look options") */
function applyEditStyle(style) {
  $("#editStyle").value = style;
  for (const k of FX) $("#fx_" + k).checked = EDIT_PRESETS[style].includes(k);
  if (style !== "off") { $("#captions").checked = true; $("#capStyle").value = "pop"; }
  showEditStyle();
}
function showEditStyle() {
  const style = $("#editStyle").value;
  $$("#editSeg button").forEach(b => b.classList.toggle("on", b.dataset.edit === style));
  $("#editNote").textContent = EDIT_NOTES[style];
  $("#fxRow").classList.toggle("hidden", style === "off");
}
function saveForm() {
  const o = { brand: 1 }; for (const id of REMEMBER) { const e = $("#" + id); o[id] = e.type === "checkbox" ? e.checked : e.value; }
  const len = $("#lenSeg .on"); o.len = len ? len.dataset.min : null; store.set("form", JSON.stringify(o));
}
function loadForm() {
  try {
    const o = JSON.parse(store.get("form") || "{}");
    for (const id of REMEMBER) {
      const e = $("#" + id); if (!e || !(id in o)) continue;
      if (e.type === "checkbox") e.checked = !!o[id];
      else if (e.tagName === "INPUT" || [...e.options].some(op => op.value === o[id])) e.value = o[id];
    }
    if (o.len) $$("#lenSeg button").forEach(b => b.classList.toggle("on", b.dataset.min === o.len));
    if (!("editStyle" in o) || !("fx_memes" in o)) applyEditStyle(o.editStyle || "viral");     // first run, or choices saved before memes
    if (!o.brand && $("#fx_cta")) $("#fx_cta").checked = false;                              // the follow card is no longer on by itself
  } catch { applyEditStyle("viral"); }
  showEditStyle();
}
document.addEventListener("change", e => { if (REMEMBER.includes(e.target.id)) saveForm(); });
$("#watermark").addEventListener("input", debounce(saveForm, 400));
$("#lenSeg").addEventListener("click", saveForm);
loadForm();
$("#faceBg").addEventListener("change", async () => {
  try { await post("/api/settings", { facecam_bg: $("#faceBg").value }); toast("Saved: new renders use this background.", "good"); }
  catch (err) { toast(err.message, "bad"); }
});
$("#memeOpen").addEventListener("click", async () => {
  try { await post("/api/memes/open"); } catch (err) { toast(err.message, "bad"); }
});
window.addEventListener("focus", () => { if (state.cfg) loadConfig().catch(() => {}); });     // back from the folder: count again
// A saved setting (not a per-job choice): Autopilot, rendering and uploads all follow it.
$("#capScript").addEventListener("change", async () => {
  try {
    await post("/api/settings", { caption_script: $("#capScript").value });
    toast("Saved. New renders use " + ($("#capScript").value === "roman" ? "Roman letters." : "the spoken script."), "good");
  } catch (err) { toast(err.message, "bad"); }
});
$("#editSeg").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { applyEditStyle(b.dataset.edit); saveForm(); syncAutopilot(); } });
$("#editStyle").addEventListener("change", () => { applyEditStyle($("#editStyle").value); saveForm(); syncAutopilot(); });

function parseLinks(text) {
  const urls = text.match(/https?:\/\/[^\s,]+/g);
  return urls && urls.length ? urls : text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}
function analyzeBody(extra = {}) {
  const len = $("#lenSeg .on");
  return { candidates: +$("#cands").value, language: $("#lang").value, focus: $("#focus").value.trim(),
           min_seconds: +len.dataset.min, max_seconds: +len.dataset.max, use_audio: $("#useAudio").checked,
           use_replay: $("#useReplay").checked, refresh: $("#refresh").checked, ...extra };
}
/* The page can be newer than the running app (files updated while it was open): never send what that app cannot understand. */
const serverCanEdit = () => !state.cfg || (!!state.cfg.edit_presets && !!state.cfg.memes);
function renderOptions() {
  const [aspect] = $("#preset").value.split("|"), edit = serverCanEdit() ? $("#editStyle").value : "off";
  const effects = { captions: $("#captions").checked };
  for (const k of FX) effects[k] = $("#fx_" + k).checked;
  const style = $("#capStyle").value;
  const o = { aspect_ratio: aspect, layout: $("#layout").value, quality: +$("#quality").value, captions: $("#captions").checked,
              caption_style: !serverCanEdit() && style === "pop" ? "classic" : style, title_card: $("#titleCard").checked,
              watermark: $("#watermark").value.trim(), normalize_audio: $("#normalize").checked, kit: $("#kit").checked };
  if (serverCanEdit()) { o.edit = edit; if (edit !== "off") o.effects = effects; }
  return o;
}

/* autopilot: link in, best clips rendered and uploaded to the channel, nobody has to click again */
const autopilotOn = () => $("#modeAuto").checked;
function syncAutopilot() {
  const on = autopilotOn(), y = state.yt || {};
  $("#modeAutoBox").classList.toggle("on", on); $("#modeManualBox").classList.toggle("on", !on);
  $("#apOpts").classList.toggle("hidden", !on);
  $("#analyze").textContent = on ? "🚀 Run autopilot" : $("#analyze").dataset.label;
  const r = renderOptions(), shape = { "9:16": "vertical 9:16", "1:1": "square", "4:5": "portrait 4:5" }[r.aspect_ratio] || r.aspect_ratio;
  const channel = y.connected && y.channel ? `“${y.channel.title}”` : "your channel";
  const editing = r.edit === "off" ? (r.captions ? ", captions (" + r.caption_style + ")" : "") : `, ${r.edit === "viral" ? "✨ viral" : "clean"} editing`;
  $("#apNote").textContent = `Clips: ${shape}, ${r.layout} layout, ${r.quality}p${editing}. ` +
    `The AI writes each title, description and hashtags. Uploads go to ${channel}` +
    (y.connected ? ` (${y.uploads_left} of ${y.daily_limit} uploads left today; extra ones wait and continue by themselves).` : ".") +
    " A moment that is already on your channel is never repeated.";
}
function autopilotBody() {
  const when = $("#apWhen").value, up = { privacy: $("#apPrivacy").value };
  if (when !== "now") { up.start_in_hours = 1; up.every_hours = +when; }      // the first one goes live an hour after the clips are ready
  return { auto_render: +$("#apCount").value, auto_min_score: +$("#apMin").value, render: { ...renderOptions(), kit: true }, upload: up };
}
$$("input[name=mode]").forEach(r => r.addEventListener("change", () => {
  const y = state.yt || {};
  if (autopilotOn() && !y.connected) {                                        // Autopilot needs a connected channel first
    $("#modeManual").checked = true; syncAutopilot();
    toast(y.needs_reconnect ? "The YouTube sign-in expired. Reconnect it first." : "Connect your YouTube channel first, then Autopilot works.", "bad");
    go("youtube"); return;
  }
  syncAutopilot();
}));
for (const id of ["apCount", "apMin", "apPrivacy", "apWhen", "preset", "layout", "quality", "captions", "capStyle"]) $("#" + id).addEventListener("change", syncAutopilot);

$("#analyze").addEventListener("click", async () => {
  const urls = parseLinks($("#urls").value);
  if (!urls.length) { $("#urls").focus(); toast("Paste a video link first."); return; }
  if (!aiKeySet()) { toast("Add your AI key first.", "bad"); openSettings("ai"); return; }
  const auto = +$("#autoRender").value, ap = autopilotOn();
  if (ap && $("#apPrivacy").value === "public") {
    const ch = state.yt && state.yt.channel ? state.yt.channel.title : "your channel";
    if (!confirm(`Autopilot will publish up to ${$("#apCount").value} clip(s) PUBLICLY on “${ch}” without asking again. Continue?`)) return;
  }
  const btn = $("#analyze"); btn.disabled = true;
  try {
    const body = ap ? analyzeBody({ urls, ...autopilotBody() }) : analyzeBody({ urls, auto_render: auto, render: auto ? renderOptions() : undefined });
    const res = await post("/api/analyze", body);
    res.jobs.forEach(trackJob);
    $("#urls").value = ""; $("#urls").rows = 1;
    if (urls.length > 1) toast(`${urls.length} videos queued. They run one after another.`);
    else if (ap) toast("Autopilot started. You can leave this page open and do something else.", "good");
    else toast("Working on it. The progress is shown above.", "good");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) { toast(e.message, "bad"); } finally { btn.disabled = false; }
});

/* ---------- jobs ---------- */
const STAGE_NAMES = { source: "Reading the link", transcript: "Transcript", audio: "Audio", signals: "Loudness", rank: "Ranking moments",
                      fetch: "Fetching the clip", render: "Rendering", kit: "Titles and hashtags", upload: "Uploading to YouTube" };
function trackJob(job) { state.jobs.set(job.id, job); renderJobs(); if (!state.poll) state.poll = setInterval(pollJobs, 1500); }
async function pollJobs() {
  const active = [...state.jobs.values()].filter(j => ["queued", "running"].includes(j.status));
  if (!active.length) { clearInterval(state.poll); state.poll = null; return; }
  for (const j of active) {
    try {
      const nj = await api("/api/jobs/" + j.id); state.jobs.set(j.id, nj);
      if (["done", "error", "cancelled"].includes(nj.status)) await onJobFinished(nj);
    } catch { /* server busy: try again next tick */ }
  }
  renderJobs();
}
async function onJobFinished(job) {
  if (job.status === "error") { toast(job.error || "That did not work.", "bad"); loadConfig().catch(() => {}); return; }
  if (job.status === "cancelled") { toast("Cancelled."); return; }
  if (job.kind === "update") {
    const r = job.result || {};
    toast(r.changed ? `yt-dlp updated: ${r.before} → ${r.after}` : "yt-dlp is already the newest version.", "good");
    if ($("#settings").open) loadHealth(false);
    return;
  }
  setTimeout(renderJobs, 46000);                                              // a finished job card tidies itself away
  const pid = job.result && job.result.project_id; if (!pid) return;
  loadLibrary(); loadConfig().catch(() => {}); if (currentIs("youtube")) loadUploads();
  const r = job.result, sent = r.uploaded != null;
  const summary = !sent ? "" : [`${r.uploaded} uploaded to YouTube`, r.failed ? `${r.failed} refused` : "",
    r.waiting ? `${r.waiting} waiting (${r.reason || "YouTube asked to wait"}); they continue by themselves` : ""].filter(Boolean).join(", ") + ".";
  if (job.kind === "upload") {
    if (state.project && state.project.id === pid) await refreshProject();
    toast(summary, r.failed || r.waiting ? "" : "good"); return;
  }
  if (job.kind === "analyze") {
    await openProject(pid, sent ? "clips" : "moments"); go("project/" + encodeURIComponent(pid));
    toast(sent ? "Autopilot finished: " + summary : "Found the best moments. Tick the ones you like.", "good");
  } else if (state.project && state.project.id === pid) {
    await refreshProject();
    if (job.kind === "render") { setTab("clips"); toast("Clips are ready. Upload them to YouTube from here, or download them.", "good"); }
    else toast("Titles and hashtags are ready.", "good");
  }
}
function jobCard(j) {
  const finished = j.status === "done", failed = j.status === "error" || j.status === "interrupted";
  const over = finished || failed || j.status === "cancelled";
  const waitingRetry = j.status === "queued" && j.retry_at;
  const names = [...new Set(j.stages.map(s => STAGE_NAMES[s] || s))];
  const cur = STAGE_NAMES[j.stage] || j.stage;
  const steps = names.map(n => el("span", { class: "step " + (over || n !== cur ? "done" : "on"), text: n }));
  let eta = "";
  if (j.status === "running" && j.fraction > 0.03 && j.started) {
    const el_s = Date.now() / 1000 - j.started, left = el_s * (1 - j.fraction) / j.fraction;
    if (left > 5) eta = ` · about ${left > 90 ? Math.round(left / 60) + " min" : Math.round(left) + " s"} left`;
  }
  const pct = finished ? 100 : (j.fraction != null ? Math.round(j.fraction * 100) : null);
  const bar = el("div", { class: "bar" + (pct == null && !over ? " indeterminate" : "") }, el("i", { style: pct != null ? `width:${Math.max(2, pct)}%` : "" }));
  let line = j.note || "";
  if (failed) line = j.error;
  else if (finished) line = "Done.";
  else if (j.status === "cancelled") line = "Cancelled.";
  else if (waitingRetry) { const s = Math.max(0, Math.round(j.retry_at - Date.now() / 1000)); line = `The AI service is busy or unreachable. Next try in ${s > 90 ? Math.round(s / 60) + " min" : s + " s"} (attempt ${j.attempt + 1} of ${j.max_attempts}). Nothing is lost.`; }
  else if (j.status === "queued") line = "Waiting for the job before it…";
  else if (!line && j.log.length) line = j.log[j.log.length - 1];
  const log = el("pre", { class: "log", text: j.log.join("\n") });
  const details = el("details", { ontoggle: e => e.target.open ? state.logOpen.add(j.id) : state.logOpen.delete(j.id) },
    el("summary", { class: "muted small", text: "Details" }), log);
  if (state.logOpen.has(j.id)) { details.open = true; setTimeout(() => { log.scrollTop = log.scrollHeight; }); }
  return el("section", { class: "panel job" + (failed ? " failed" : over ? " finished" : "") },
    el("div", { class: "head" },
      el("b", { text: ({ analyze: "Finding moments: ", render: "Rendering: ", kit: "Writing text: ", upload: "Uploading: ", update: "" }[j.kind] || "") + j.label }),
      el("div", { class: "grow" },
        failed && j.request ? el("button", { class: "small primary", title: "Sends exactly the same request again. Everything already finished is saved, so it continues from there.",
          onclick: () => retryJob(j), text: "Try again" }) : null,
        failed ? el("button", { class: "small", title: "Copies the error and the log, to paste into a bug report",
          onclick: () => copyText(`Shorts Studio ${state.cfg ? state.cfg.version : ""}\n${j.kind}: ${j.label}\n${j.error || ""}\n\n${j.log.join("\n")}`, "Details copied"), text: "Copy details" }) : null,
        over ? el("button", { class: "small", onclick: () => dismiss(j.id), text: "Dismiss" })
             : el("button", { class: "small danger", onclick: () => cancelJob(j.id), text: "Cancel" }))),
    el("div", { class: "steps" }, steps),
    over && !failed ? null : (waitingRetry ? null : bar),
    el("p", { class: "small " + (failed ? "err" : "muted"), style: "margin:8px 0 0", text: line + (over ? "" : eta) }),
    details);
}
function renderJobs() {
  const now = Date.now() / 1000;
  const list = [...state.jobs.values()].filter(j => !state.dismissed.has(j.id) && !(j.status === "done" && j.finished && now - j.finished > 45)).sort((a, b) => a.created - b.created);
  $("#jobs").replaceChildren(...list.map(jobCard));
  const run = list.find(j => j.status === "running");                         // the tab title shows progress when you are elsewhere
  document.title = run ? `${run.fraction != null ? "(" + Math.round(run.fraction * 100) + "%) " : "… "}Shorts Studio` : "Shorts Studio";
}
function dismiss(id) {
  state.dismissed.add(id); renderJobs();
  store.set("dismissed", JSON.stringify([...state.dismissed].slice(-80)));
}
async function retryJob(j) {
  /* everything already finished (transcript, signals, AI answers) is cached, so this picks up where it stopped */
  dismiss(j.id);
  try {
    const r = await post(j.request.path, j.request.body);          // the very request that failed, options and all
    (r.jobs || [r]).forEach(trackJob);
  } catch (e) { toast(e.message, "bad"); }
}
async function cancelJob(id) { try { state.jobs.set(id, await post(`/api/jobs/${id}/cancel`)); renderJobs(); } catch (e) { toast(e.message, "bad"); } }

/* ---------- your videos ---------- */
function projCard(p) {
  const thumb = el("div", { class: "thumb" }, p.thumbnail ? null : "no preview");
  if (p.thumbnail) thumb.style.backgroundImage = `url("${p.thumbnail}")`;
  const bits = [p.duration ? clock(p.duration) : null, plural(p.n_candidates, "moment", "moments"), plural(p.n_clips, "clip", "clips")].filter(Boolean).join(" · ");
  return el("button", { class: "proj" + (state.project && state.project.id === p.id ? " active" : ""), onclick: () => go("project/" + encodeURIComponent(p.id)) },
    thumb, el("div", { class: "meta" }, el("b", { text: p.title || p.id }), el("span", { text: bits }),
      p.n_uploaded ? el("span", { class: "up", text: " · ▶ " + p.n_uploaded + " on YouTube" }) : null,
      p.n_queued ? el("span", { text: " · ⏳ " + p.n_queued + " waiting" }) : null));
}
async function loadLibrary() {
  state.projects = await api("/api/projects");
  const list = state.projects, q = $("#projSearch").value.trim().toLowerCase();
  const shown = q ? list.filter(p => (p.title || p.id).toLowerCase().includes(q)) : list;
  $("#navProj").textContent = list.length ? String(list.length) : "";
  $("#lib").replaceChildren(...shown.map(projCard));
  $("#libEmpty").hidden = list.length > 0;
  $("#recentBox").hidden = list.length === 0; $("#recent").replaceChildren(...list.slice(0, 3).map(projCard));
  renderSetup();
  return list;
}
$("#projSearch").addEventListener("input", debounce(() => loadLibrary().catch(() => {}), 150));

async function openProject(pid, tab = "moments") {
  state.project = await api("/api/projects/" + encodeURIComponent(pid));
  state.picked = new Set(); state.edits = {}; state.players = {}; state.clipPicked = new Set(); state.shown = PAGE;
  store.set("lastProject", pid);
  renderProject(); setTab(tab); loadSignals();
}
async function refreshProject() {
  const keep = new Set(state.picked);
  state.project = await api("/api/projects/" + encodeURIComponent(state.project.id));
  state.picked = new Set([...keep].filter(i => i < state.project.candidates.length));
  renderProject(); loadSignals(); loadLibrary();
}
function setTab(name) {
  state.tab = name;
  for (const [t, pane] of [["moments", "#paneMoments"], ["clips", "#paneClips"]]) {
    $("#tab" + t[0].toUpperCase() + t.slice(1)).setAttribute("aria-selected", String(t === name)); $(pane).hidden = t !== name;
  }
  if (name === "moments") requestAnimationFrame(drawTimeline);
}
$("#tabMoments").addEventListener("click", () => setTab("moments"));
$("#tabClips").addEventListener("click", () => setTab("clips"));

/* ---------- project view ---------- */
function candTimes(c) { const e = state.edits[c.id]; return e ? { start: e.start, end: e.end } : { start: c.start_time, end: c.end_time }; }
function renderProject() {
  const p = state.project; if (!p) return;
  const src = p.source || {};
  const img = $("#pThumb"); img.hidden = !src.thumbnail; if (src.thumbnail) img.src = src.thumbnail;
  $("#pTitle").textContent = p.title || p.id;
  const chips = [el("span", { class: "chip", text: `${clock(p.duration || 0)} long` })];
  const ti = p.transcript_info || {};
  if (ti.source) chips.push(el("span", { class: "chip " + (ti.source.startsWith("youtube") ? "ok" : "warn"),
    title: ti.source, text: ti.source.startsWith("youtube") ? `YouTube captions${ti.language ? " · " + ti.language : ""}` : "Speech recognition" }));
  const sig = (p.options && p.options.signals) || [];
  if (sig.includes("energy")) chips.push(el("span", { class: "chip acc", text: "voice energy" }));
  if (sig.includes("replay")) chips.push(el("span", { class: "chip acc", text: "most replayed" }));
  if (p.options && p.options.focus) chips.push(el("span", { class: "chip", text: "focus: " + p.options.focus }));
  $("#pChips").replaceChildren(...chips);
  const watch = $("#btnWatch"); watch.hidden = src.kind === "file";
  if (src.kind !== "file") watch.href = src.kind === "youtube" ? `https://youtu.be/${src.id}` : src.url;
  $("#rFocus").value = (p.options && p.options.focus) || "";
  $("#rMin").value = (p.options && p.options.min_seconds) || 20; $("#rMax").value = Math.min((p.options && p.options.max_seconds) || 59, 59);
  $("#cntMoments").textContent = String(p.candidates.length);
  $("#cntClips").textContent = String(p.clips.filter(c => c.url || c.removed_after_upload).length || "");
  renderCards(); renderClips(); updateRenderButton();
}
function updateRenderButton() {
  const n = state.picked.size, busy = [...state.jobs.values()].some(j => j.kind === "render" && ["queued", "running"].includes(j.status));
  const b = $("#render"); b.disabled = n === 0 || busy; b.textContent = n ? `Render ${plural(n, "clip", "clips")}` : "Render selected";
  $("#kitSel").disabled = n === 0;
  $("#pickInfo").textContent = n ? `${n} selected` : "Tick the moments you like";
  const sorted = visibleCandidates(); $("#count").textContent = `${sorted.length} of ${state.project.candidates.length} shown · ${n} selected`;
}
function visibleCandidates() {
  let list = [...state.project.candidates];
  if ($("#hideWeak").checked) list = list.filter(c => c.score == null || c.score >= 50);
  if ($("#sort").value === "time") list.sort((a, b) => a.start_time - b.start_time);
  else list.sort((a, b) => (b.score == null ? -1 : b.score) - (a.score == null ? -1 : a.score) || a.id - b.id);
  return list;
}
function renderCards() {
  state.players = {};
  const list = visibleCandidates(), shown = list.slice(0, state.shown), left = list.length - shown.length;
  $("#cards").replaceChildren(...(shown.length ? shown.map(card) : [el("div", { class: "empty", text: "Nothing to show." })]));
  const more = $("#moreCards"); more.hidden = left <= 0;
  more.textContent = left > 0 ? `Show ${Math.min(PAGE, left)} more (${left} left)` : "";
  updateRenderButton(); drawTimeline();
}
$("#moreCards").addEventListener("click", () => { state.shown += PAGE; renderCards(); });
function durText(c) {
  const t = candTimes(c), d = t.end - t.start; const bad = !(t.start >= 0 && d >= 3 && d <= 180);
  return { text: d > 59.05 && !bad ? `${Math.round(d)} s → cut to 59 s` : `${Math.round(d)} s`, bad };
}
function setTimes(c, start, end, cardEl) {
  const orig = { start: c.start_time, end: c.end_time };
  const same = Math.abs(start - orig.start) < 0.05 && Math.abs(end - orig.end) < 0.05;
  if (same) delete state.edits[c.id]; else state.edits[c.id] = { start, end };
  if (cardEl) refreshTrim(c, cardEl);
}
function refreshTrim(c, cardEl) {
  const t = candTimes(c), d = durText(c);
  $(".t-start", cardEl).value = clock(t.start); $(".t-end", cardEl).value = clock(t.end);
  const dur = $(".dur", cardEl); dur.textContent = d.text; dur.classList.toggle("bad", d.bad);
  $(".edited", cardEl).classList.toggle("hidden", !state.edits[c.id]);
  $(".reset", cardEl).classList.toggle("hidden", !state.edits[c.id]);
}
function card(c) {
  const p = state.project, src = p.source || {}, picked = state.picked.has(c.id);
  const rendered = p.clips.filter(x => x.candidate === c.id && x.url);
  const sc = c.score == null ? "lo" : c.score >= 85 ? "hi" : c.score >= 65 ? "mid" : "lo";
  const sg = c.signals || {};
  const badges = [];
  if (c.manual) badges.push(el("span", { class: "badge mine", text: "added by you" }));
  if (c.llm_score != null) badges.push(el("span", { class: "badge", title: "How strongly the AI editor rated this moment", text: `AI ${c.llm_score}` }));
  if (sg.energy != null) badges.push(el("span", { class: "badge" + (sg.energy < 30 ? " quiet" : ""), title: "How loud and excited the voice is compared with the rest of the video",
    text: `Voice ${sg.energy}` + (sg.energy < 30 ? " · quiet" : "") }));
  if (sg.replay != null) badges.push(el("span", { class: "badge", title: "How much viewers re-watched this part", text: `Replays ${sg.replay}` }));
  badges.push(el("span", { class: "badge edited hidden", text: "trimmed" }));

  const cardEl = el("article", { class: "card" + (picked ? " picked" : ""), id: "card-" + c.id });
  const chk = el("input", { type: "checkbox", id: "pick" + c.id, checked: picked || null, onchange: e => {
    e.target.checked ? state.picked.add(c.id) : state.picked.delete(c.id);
    cardEl.classList.toggle("picked", e.target.checked); updateRenderButton(); drawTimeline(); } });

  const tIn = cls => el("input", { type: "text", class: cls, "aria-label": cls === "t-start" ? "Start time" : "End time" });
  const tStart = tIn("t-start"), tEnd = tIn("t-end");
  const apply = () => {
    const s = parseTime(tStart.value), e = parseTime(tEnd.value);
    if (Number.isNaN(s) || Number.isNaN(e)) { refreshTrim(c, cardEl); return; }
    setTimes(c, s, e, cardEl);
  };
  tStart.addEventListener("change", apply); tEnd.addEventListener("change", apply);
  const nudge = (which, d) => () => { const t = candTimes(c); const s = which === "s" ? Math.max(0, t.start + d) : t.start, e = which === "e" ? t.end + d : t.end; setTimes(c, s, e, cardEl); };
  const trim = el("div", { class: "trim" },
    el("button", { class: "small", title: "1 second earlier", onclick: nudge("s", -1), text: "−" }), tStart,
    el("button", { class: "small", title: "1 second later", onclick: nudge("s", 1), text: "+" }),
    el("span", { class: "muted", text: "to" }),
    el("button", { class: "small", title: "1 second earlier", onclick: nudge("e", -1), text: "−" }), tEnd,
    el("button", { class: "small", title: "1 second later", onclick: nudge("e", 1), text: "+" }),
    el("button", { class: "small reset hidden", title: "Back to the suggested times", onclick: () => { delete state.edits[c.id]; refreshTrim(c, cardEl); }, text: "↺" }),
    el("span", { class: "dur", text: "" }));

  const previewBox = el("div", { class: "preview" });
  const setRow = el("div", { class: "setrow hidden" },
    el("button", { class: "small", onclick: () => { const t = playerTime(c); if (t != null) setTimes(c, t, Math.max(t + 3, candTimes(c).end), cardEl); }, text: "Start = now" }),
    el("button", { class: "small", onclick: () => { const t = playerTime(c); if (t != null) setTimes(c, candTimes(c).start, t, cardEl); }, text: "End = now" }),
    el("button", { class: "small", onclick: () => replay(c), text: "↻ Play this range" }));
  const startPreview = () => preview(c, previewBox, setRow);
  if (src.kind === "youtube" && src.embeddable) previewBox.append(el("button", { onclick: startPreview, text: "▶ Preview this moment" }));
  else if (src.local_available) previewBox.append(el("button", { onclick: startPreview, text: "▶ Preview this moment" }));
  else previewBox.append(el("a", { href: src.kind === "youtube" ? `https://youtu.be/${src.id}?t=${Math.floor(c.start_time)}` : src.url, target: "_blank", rel: "noopener",
    text: "Open the video at this moment ↗" }));

  const kit = c.kit ? el("div", { class: "kit" },
    kitLine("Title", c.kit.title), kitLine("Description", c.kit.description), kitLine("Hashtags", (c.kit.hashtags || []).join(" "), "tags"),
    el("button", { class: "small", onclick: () => copyText(`${c.kit.title}\n\n${c.kit.description}\n\n${(c.kit.hashtags || []).join(" ")}`, "Post text copied"), text: "Copy everything" })) : null;

  cardEl.append(...[
    el("div", { class: "top" }, el("span", { class: "rank", text: `#${c.id + 1}` }), el("span", { class: `score ${sc}`, text: c.score == null ? "—" : c.score }),
      el("label", { class: "check pick", for: "pick" + c.id }, chk, "Pick")),
    el("div", { class: "badges" }, badges),
    el("h3", { text: c.title || "Untitled" }),
    trim,
    c.hook_sentence ? el("p", { class: "hook", text: c.hook_sentence }) : null,
    previewBox, setRow,
    el("details", { class: "sub" }, el("summary", { text: "Why it was picked · what is said" }),
      c.rerank_reason || c.virality_reason ? el("p", { text: c.rerank_reason || c.virality_reason }) : null,
      c.excerpt ? el("p", { text: "“" + c.excerpt + "”" }) : null),
    kit,
    rendered.length ? el("div", { class: "done-links" }, rendered.map(r => el("a", { href: r.url, target: "_blank", rel: "noopener",
      text: `✓ rendered (${r.layout}${r.captions ? ", captions" : ""})` }))) : null].filter(Boolean));   // never append a bare null: it would print "null"
  refreshTrim(c, cardEl);
  return cardEl;
}
function kitLine(label, value, cls = "") {
  return el("div", { class: "line" }, el("b", { class: "small muted", text: label }), el("span", { class: cls, text: value }),
    el("button", { class: "small", title: "Copy", onclick: () => copyText(value, label + " copied"), text: "⧉" }));
}

/* preview: the video's own site (YouTube player API) or the local file */
let ytReady = null;
function loadYT() {
  if (ytReady) return ytReady;
  ytReady = new Promise((resolve, reject) => {
    if (window.YT && window.YT.Player) return resolve();
    const s = document.createElement("script"); s.src = "https://www.youtube.com/iframe_api"; s.onerror = () => { ytReady = null; reject(new Error("The YouTube player could not load (are you offline?)")); };
    window.onYouTubeIframeAPIReady = resolve; document.head.append(s);
  });
  return ytReady;
}
async function preview(c, box, setRow) {
  const src = state.project.source, t = candTimes(c);
  if (src.kind === "youtube" && src.embeddable) {
    try { await loadYT(); } catch (e) { toast(e.message, "bad"); return; }
    const holder = el("div"); box.replaceChildren(holder);
    state.players[c.id] = new YT.Player(holder, { videoId: src.id, width: "100%", height: "100%",
      playerVars: { start: Math.floor(t.start), end: Math.ceil(t.end), autoplay: 1, rel: 0, modestbranding: 1, playsinline: 1 } });
    setRow.classList.remove("hidden");
  } else if (src.local_available) {
    box.replaceChildren(el("video", { controls: true, autoplay: true, preload: "metadata", src: `${state.project.source_media}#t=${t.start.toFixed(2)},${t.end.toFixed(2)}` }));
  }
}
function playerTime(c) {
  const pl = state.players[c.id];
  if (!pl || !pl.getCurrentTime) { toast("Start the preview first, then pause where you want."); return null; }
  return Math.max(0, Math.round(pl.getCurrentTime() * 10) / 10);
}
function replay(c) {
  const pl = state.players[c.id], t = candTimes(c);
  if (pl && pl.loadVideoById) pl.loadVideoById({ videoId: state.project.source.id, startSeconds: t.start, endSeconds: t.end });
}

/* ---------- rendered clips + uploading ---------- */
const UP_BUSY = ["queued", "uploading"];
const ytState = c => (c.youtube && c.youtube.state) || "";
const canPick = c => c.url && !UP_BUSY.includes(ytState(c)) && ytState(c) !== "uploaded";

function ytLine(c) {
  const y = c.youtube; if (!y) return null;
  if (y.state === "uploaded") {
    return el("div", { class: "ytline ok" },
      el("a", { href: y.url, target: "_blank", rel: "noopener", text: "✓ On YouTube" }),
      el("span", { class: "badge", text: y.publish_at ? "goes live " + localTime(y.publish_at) : (y.actual_privacy || y.privacy || "") }),
      y.studio_url ? el("a", { href: y.studio_url, target: "_blank", rel: "noopener", text: "Studio" }) : null,
      el("button", { class: "small", title: "Upload this clip once more as a new video", onclick: () => openYtUpload([c.file], true), text: "Upload again" }),
      y.notice ? el("p", { class: "notice", text: y.notice }) : null);
  }
  if (y.state === "uploading") return el("div", { class: "ytline" }, "⬆ Uploading to YouTube…");
  if (y.state === "queued") {
    return el("div", { class: "ytline" }, "⏳ Waiting to upload" + (y.publish_at ? " (goes live " + localTime(y.publish_at) + ")" : ""),
      el("button", { class: "small", onclick: () => dequeueClips([c.file]), text: "Remove from queue" }),
      y.error ? el("p", { class: "muted", text: "Last try: " + y.error }) : null);
  }
  if (y.state === "failed") {
    return el("div", { class: "ytline bad" }, el("span", { class: "err", text: "✗ Not uploaded" }),
      el("button", { class: "small", onclick: () => retryClipUpload(c.file, y), text: "Try again" }),
      el("p", { class: "err", text: y.error || "YouTube refused this clip." }));
  }
  return null;
}

function renderClips() {
  const all = state.project.clips.filter(c => c.url || c.removed_after_upload).sort((a, b) => (b.created || "").localeCompare(a.created || ""));
  const clips = all.filter(c => c.url), posted = all.filter(c => !c.url);
  const failed = state.project.clips.filter(c => c.error);
  state.clipPicked = new Set([...state.clipPicked].filter(f => clips.some(c => c.file === f && canPick(c))));
  const nodes = clips.map(c => {
    const cand = c.candidate != null ? state.project.candidates[c.candidate] : null;
    const kit = cand && cand.kit;
    const shape = c.aspect_ratio === "1:1" ? " square" : c.aspect_ratio === "4:5" ? " portrait" : "";
    const sum = c.edit_summary, edited = c.edit && c.edit !== "off"
      ? `✨ ${c.edit}${sum ? ` (${[sum.memes ? plural(sum.memes, "meme", "memes") : null, sum.popups ? plural(sum.popups, "pop-up", "pop-ups") : null, sum.zooms ? plural(sum.zooms, "zoom", "zooms") : null,
          sum.removed_seconds >= 0.3 ? sum.removed_seconds + "s of pauses cut" : null].filter(Boolean).join(", ")})` : ""}` : null;
    const flags = [edited, c.framing || c.layout, !edited && c.captions ? `captions (${c.caption_style || "classic"})` : null, c.title_card ? "title" : null, c.quality ? c.quality + "p" : null].filter(Boolean).join(" · ");
    const picked = state.clipPicked.has(c.file);
    const chk = el("input", { type: "checkbox", checked: picked || null, disabled: !canPick(c) || null, "aria-label": "Select this clip for upload",
      onchange: e => { e.target.checked ? state.clipPicked.add(c.file) : state.clipPicked.delete(c.file); e.target.closest(".clip").classList.toggle("picked", e.target.checked); renderYtBar(); } });
    return el("div", { class: "clip" + shape + (picked ? " picked" : "") },
      el("label", { class: "clippick" }, chk, canPick(c) ? "Select for YouTube" : ytState(c) === "uploaded" ? "Already uploaded" : "In the upload queue"),
      el("video", { controls: true, preload: "metadata", src: c.url }),
      el("div", { class: "small" }, el("b", { text: (c.candidate != null ? `#${c.candidate + 1} ` : "") + (c.title || (cand && cand.title) || "") }),
        el("div", { class: "muted", text: `${flags} · ${clock(c.start || 0)}–${clock(c.end || 0)}` })),
      ytLine(c),
      el("div", { class: "row" },
        el("a", { href: c.url, download: c.file }, el("button", { class: "small", type: "button", text: "Download" })),
        kit ? el("button", { class: "small", onclick: () => copyText(`${kit.title}\n\n${kit.description}\n\n${(kit.hashtags || []).join(" ")}`, "Post text copied"), text: "Copy title + hashtags" }) : null,
        el("button", { class: "small danger", title: "Delete this clip's file", onclick: () => deleteClip(c), text: "Delete" })),
      kit ? el("div", { class: "small muted", text: kit.title }) : null);
  });
  posted.forEach(c => nodes.push(el("div", { class: "clip" }, el("b", { class: "small", text: (c.candidate != null ? `#${c.candidate + 1} ` : "") + (c.title || "") }),
    el("p", { class: "small" }, "✓ Posted on YouTube. The file was removed from this computer to save space."),
    c.youtube_url ? el("a", { href: c.youtube_url, target: "_blank", rel: "noopener", text: "Watch on YouTube ↗" }) : null)));
  const zip = $("#btnZip"); zip.hidden = clips.length === 0; zip.href = `/api/projects/${ytId()}/export.zip`;
  failed.forEach(c => nodes.push(el("div", { class: "clip" }, el("b", { class: "small", text: `#${(c.candidate ?? 0) + 1} ${c.title || ""}` }),
    el("p", { class: "err small", text: "Could not render: " + c.error }))));
  $("#clips").replaceChildren(...(nodes.length ? nodes : [el("div", { class: "empty" }, "Nothing rendered yet. On the ",
    el("a", { href: "#", onclick: e => { e.preventDefault(); setTab("moments"); }, text: "first tab" }), " tick the moments you like, then press “Render”.")]));
  renderYtBar(); scheduleYtWatch();
}

async function deleteClip(c) {
  if (!confirm(`Delete “${c.title || c.file}”? The file is removed from the disk.`)) return;
  try {
    await api(`/api/projects/${ytId()}/clips/${encodeURIComponent(c.file)}`, { method: "DELETE" });
    await refreshProject(); toast("Clip deleted.");
  } catch (e) { toast(e.message, "bad"); }
}

function renderYtBar() {
  const p = state.project; if (!p) return;
  const clips = p.clips.filter(c => c.url), y = state.yt || {};
  const pickable = clips.filter(canPick), n = state.clipPicked.size;
  const queued = clips.filter(c => ytState(c) === "queued").length, going = clips.filter(c => ytState(c) === "uploading").length;
  $("#ytSelAll").hidden = $("#ytSelNone").hidden = $("#ytUploadSel").hidden = clips.length === 0;
  $("#ytSelAll").disabled = pickable.length === 0;
  const up = $("#ytUploadSel"); up.disabled = n === 0; up.textContent = n ? `⬆ Upload ${n} to YouTube` : "⬆ Upload to YouTube";
  $("#ytDequeue").hidden = queued === 0;
  const note = $("#ytQueueNote"), bits = [];
  if (going) bits.push(`${going} uploading now`);
  if (queued) bits.push(`${queued} waiting to upload`);
  if (queued && y.paused) bits.push("uploads are paused: press Resume on the YouTube page");
  else if (queued && y.parked_until) bits.push(`YouTube asked to wait: uploads continue ${localTime(y.parked_until * 1000)}`);
  else if (queued && y.connected && y.uploads_left === 0) bits.push(`today's limit of ${y.daily_limit} uploads is reached: they continue tomorrow`);
  else if (queued && y.connected && !y.auto_resume) bits.push("automatic continuing is switched off in the YouTube settings");
  else if (queued && !y.connected) bits.push("connect YouTube so they can go");
  note.classList.toggle("hidden", bits.length === 0); note.textContent = bits.join(" · ");
}
$("#ytSelAll").addEventListener("click", () => { state.clipPicked = new Set(state.project.clips.filter(canPick).map(c => c.file)); renderClips(); });
$("#ytSelNone").addEventListener("click", () => { state.clipPicked = new Set(); renderClips(); });
$("#ytUploadSel").addEventListener("click", () => openYtUpload([...state.clipPicked]));
$("#ytDequeue").addEventListener("click", () => dequeueClips(null));

async function dequeueClips(files) {
  try {
    const r = await post(`/api/projects/${ytId()}/upload/dequeue`, files ? { files } : { all: true });
    toast(r.removed ? "Taken out of the upload queue." : "Nothing was waiting.", r.removed ? "good" : ""); await refreshProject(); loadConfig();
  } catch (e) { toast(e.message, "bad"); }
}
async function retryClipUpload(file, y) {
  try {
    const r = await post(`/api/projects/${ytId()}/upload`, { files: [file], privacy: y.privacy || undefined });
    if (r.job) trackJob(r.job);
    await refreshProject();
  } catch (e) { toast(e.message, "bad"); }
}
function yuItem(c, file) {
  const cand = c.candidate != null ? state.project.candidates[c.candidate] : null, kit = (cand && cand.kit) || {};
  const init = { title: kit.title || c.title || (cand && cand.title) || "", description: kit.description || "", hashtags: (kit.hashtags || []).join(" ") };
  const t = el("input", { type: "text", maxlength: "100", value: init.title, "aria-label": "Title" });
  const d = el("textarea", { rows: "2", "aria-label": "Description", placeholder: kit.title ? "" : "The AI writes this when it uploads, or type your own" }); d.value = init.description;
  const h = el("input", { type: "text", value: init.hashtags, "aria-label": "Hashtags", placeholder: "#funny #gaming" });
  const box = el("div", { class: "yui", "data-file": file }, el("b", { class: "small", text: `${c.candidate != null ? "#" + (c.candidate + 1) + " " : ""}${clock(c.start || 0)}–${clock(c.end || 0)}` }), t, d, h);
  box.collect = () => {
    const o = {}; if (t.value.trim() && t.value.trim() !== init.title) o.title = t.value.trim();
    if (d.value.trim() !== init.description.trim()) o.description = d.value.trim();
    if (h.value.trim() !== init.hashtags.trim()) o.hashtags = h.value.trim();
    return o;
  };
  return box;
}
function openYtUpload(files, force = false) {
  const y = state.yt || {};
  if (!y.connected) { toast(y.needs_reconnect ? "The YouTube sign-in expired. Reconnect it first." : "Connect your YouTube channel first.", "bad"); go("youtube"); return; }
  const dlg = $("#ytUpload"); dlg.dataset.files = JSON.stringify(files); dlg.dataset.force = force ? "1" : "";
  const channel = (y.channel && y.channel.title) || "your channel";
  $("#yuSummary").textContent = `${plural(files.length, "clip", "clips")} will be uploaded to “${channel}” as YouTube Shorts. ` +
    (y.uploads_left >= files.length ? "" : `Only ${y.uploads_left} more upload${y.uploads_left === 1 ? "" : "s"} fit into today's limit of ${y.daily_limit}; the rest waits and continues by itself.`) +
    (y.paused ? " Uploads are paused right now, so they will wait until you resume." : "");
  $("#yuPrivacy").value = y.privacy || "private"; $("#yuWhen").value = "now"; $("#yuErr").textContent = "";
  const t = new Date(Date.now() + 24 * 3600e3); t.setHours(18, 0, 0, 0);
  $("#yuStart").value = new Date(t - t.getTimezoneOffset() * 60e3).toISOString().slice(0, 16);
  const clips = files.map(f => state.project.clips.find(c => c.file === f)).filter(Boolean);
  $("#yuItems").replaceChildren(...clips.map(c => yuItem(c, c.file)));
  $("#yuReview").open = false;
  const warn = clips.flatMap(c => {
    const [w, h] = String(c.aspect_ratio || "9:16").split(":").map(Number), len = (c.end || 0) - (c.start || 0), out = [];
    if (w > h) out.push(`“${c.title || c.file}” is wide (${c.aspect_ratio}): YouTube will not treat it as a Short.`);
    if (len > 60) out.push(`“${c.title || c.file}” is ${Math.round(len)} s long: render it again to bring it under a minute.`);
    return out;
  });
  $("#yuWarn").replaceChildren(...warn.map(w => el("p", { text: w })));
  syncYtWhen(); dlg.showModal();
}
function syncYtWhen() {
  const sched = $("#yuWhen").value === "schedule";
  $("#yuStartWrap").classList.toggle("hidden", !sched); $("#yuEveryWrap").classList.toggle("hidden", !sched || JSON.parse($("#ytUpload").dataset.files || "[]").length < 2);
  $("#yuPrivacy").disabled = sched;
  $("#yuNote").textContent = sched ? "Scheduled videos are uploaded as private and YouTube makes them public by itself at the chosen time. Your computer can be off by then."
    : $("#yuPrivacy").value === "public" ? "Public videos are visible to everyone as soon as they are processed." : "";
}
$("#yuWhen").addEventListener("change", syncYtWhen); $("#yuPrivacy").addEventListener("change", syncYtWhen);
$("#yuCancel").addEventListener("click", () => $("#ytUpload").close());
$("#yuGo").addEventListener("click", async () => {
  const dlg = $("#ytUpload"), files = JSON.parse(dlg.dataset.files || "[]"), sched = $("#yuWhen").value === "schedule";
  const body = { files, privacy: $("#yuPrivacy").value, force: !!dlg.dataset.force };
  const meta = {}; $$("#yuItems .yui").forEach(b => { const o = b.collect(); if (Object.keys(o).length) meta[b.dataset.file] = o; });
  if (Object.keys(meta).length) body.meta = meta;
  if (sched) {
    const start = new Date($("#yuStart").value).getTime();
    if (!(start > Date.now() + 11 * 60e3)) { $("#yuErr").textContent = "Pick a time at least 10 minutes from now."; return; }
    body.schedule_start = start / 1000; body.every_hours = +$("#yuEvery").value;
  }
  $("#yuGo").disabled = true; $("#yuErr").textContent = "";
  try {
    const r = await post(`/api/projects/${ytId()}/upload`, body);
    dlg.close(); state.clipPicked = new Set();
    if (r.job) trackJob(r.job);
    if (r.skipped.length) toast(r.skipped.map(s => s.reason).filter((v, i, a) => a.indexOf(v) === i).join(" "), r.job ? "" : "bad");
    await refreshProject(); loadConfig();
  } catch (e) { $("#yuErr").textContent = e.message; } finally { $("#yuGo").disabled = false; }
});
/* clips waiting for YouTube change by themselves (the app uploads in the background): look every few seconds, redraw only on change */
function scheduleYtWatch() {
  const busy = state.project && state.project.clips.some(c => UP_BUSY.includes(ytState(c)));
  if (!busy) { clearInterval(state.ytTimer); state.ytTimer = null; return; }
  if (state.ytTimer) return;
  state.ytTimer = setInterval(async () => {
    const p = state.project; if (!p) return;
    try {
      for (const j of await api("/api/jobs")) if (["queued", "running"].includes(j.status) && !state.jobs.has(j.id)) trackJob(await api("/api/jobs/" + j.id));   // started by the background watcher
      const fresh = await api("/api/projects/" + encodeURIComponent(p.id)), sig = x => JSON.stringify(x.clips.map(c => [c.file, c.youtube && c.youtube.state, c.youtube && c.youtube.video_id]));
      if (sig(fresh) !== sig(p)) { state.project = fresh; renderClips(); loadConfig().catch(() => {}); }
    } catch { /* busy: next time */ }
  }, 8000);
}

/* ---------- timeline ---------- */
async function loadSignals() {
  try { state.signals = await api(`/api/projects/${ytId()}/signals`); } catch { state.signals = null; }
  drawTimeline();
}
function drawTimeline() {
  const cv = $("#tl"), s = state.signals, p = state.project; if (!cv || !s || !p || $("#project").hidden || $("#paneMoments").hidden) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W) return;
  cv.width = W * dpr; cv.height = H * dpr; const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const dur = s.duration || p.duration || 1, xOf = t => t / dur * W;
  const accent = css("--accent"), cyan = css("--cyan"), line = css("--line"), muted = css("--muted"), good = css("--good"), warn = css("--warn");
  ctx.clearRect(0, 0, W, H); ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, H - 30.5); ctx.lineTo(W, H - 30.5); ctx.stroke();
  const area = (series, color, alpha) => {
    if (!series || !series.length) return; const mx = Math.max(...series) || 1, top = 6, base = H - 32;
    ctx.beginPath(); ctx.moveTo(0, base);
    series.forEach((v, i) => ctx.lineTo(i / (series.length - 1) * W, base - (v / mx) * (base - top)));
    ctx.lineTo(W, base); ctx.closePath(); ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
  };
  area(s.replay, cyan, .35);
  if (s.energy && s.energy.length) {
    const sorted = [...s.energy].sort((a, b) => a - b), lo = sorted[Math.floor(sorted.length * .05)], hi = sorted[Math.floor(sorted.length * .99)] || lo + 1;
    ctx.beginPath(); ctx.strokeStyle = accent; ctx.lineWidth = 1.5;
    s.energy.forEach((v, i) => { const y = (H - 32) - Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))) * (H - 40), x = i / (s.energy.length - 1) * W; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  state.hits = [];
  s.candidates.forEach(c => {
    const t = candTimes(p.candidates[c.id] || { id: c.id, start_time: c.start, end_time: c.end });
    const x0 = xOf(t.start), w = Math.max(5, xOf(t.end) - x0), picked = state.picked.has(c.id);
    ctx.fillStyle = c.score == null ? accent : c.score >= 85 ? good : c.score >= 65 ? warn : muted; ctx.globalAlpha = picked ? 1 : .75;
    ctx.fillRect(x0, H - 24, w, 14); ctx.globalAlpha = 1;
    if (picked) { ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, H - 25, w + 2, 16); }
    state.hits.push({ x0, x1: x0 + w, id: c.id });
  });
  ctx.fillStyle = muted; ctx.font = "11px system-ui"; ctx.textAlign = "left"; ctx.fillText("0:00", 2, H - 1); ctx.textAlign = "right"; ctx.fillText(clock(dur), W - 2, H - 1);
  const leg = [];
  if (s.energy && s.energy.length) leg.push(el("span", {}, el("i", { style: `background:${accent}` }), "voice energy"));
  if (s.replay && s.replay.length) leg.push(el("span", {}, el("i", { style: `background:${cyan}` }), "most replayed"));
  leg.push(el("span", {}, el("i", { style: `background:${good}` }), "suggested moments (click one)"));
  $("#tlLegend").replaceChildren(...leg);
}
function hitAt(x) { return state.hits.map(h => ({ h, d: x >= h.x0 && x <= h.x1 ? 0 : Math.min(Math.abs(x - h.x0), Math.abs(x - h.x1)) })).filter(o => o.d < 7).sort((a, b) => a.d - b.d)[0]; }
$("#tl").addEventListener("click", e => {
  const hit = hitAt(e.offsetX); if (!hit) return;
  let target = $("#card-" + hit.h.id);
  if (!target) {                                               // not on the page yet (paging) or hidden by a filter
    const idx = visibleCandidates().findIndex(c => c.id === hit.h.id);
    if (idx < 0) { toast("That moment is hidden by the filter."); return; }
    state.shown = Math.ceil((idx + 1) / PAGE) * PAGE; renderCards(); target = $("#card-" + hit.h.id);
  }
  target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash");
});
$("#tl").addEventListener("mousemove", e => {
  const tip = $("#tlTip"), p = state.project, s = state.signals; if (!p || !s) return;
  const t = e.offsetX / $("#tl").clientWidth * (s.duration || p.duration), hit = hitAt(e.offsetX);
  const c = hit && p.candidates[hit.h.id];
  tip.textContent = clock(t) + (c ? `  ·  #${c.id + 1} ${c.title}${c.score != null ? " (" + c.score + ")" : ""}` : "");
  tip.style.display = "block"; tip.style.left = Math.min(e.offsetX + 14, $("#tlBox").clientWidth - 200) + "px"; tip.style.top = "-6px";
});
$("#tl").addEventListener("mouseleave", () => { $("#tlTip").style.display = "none"; });
window.addEventListener("resize", debounce(drawTimeline, 120));

/* ---------- project actions ---------- */
$("#preset").addEventListener("change", () => { /* presets only set the shape; layout stays the user's choice */ });
$("#sort").addEventListener("change", () => { state.shown = PAGE; renderCards(); }); $("#hideWeak").addEventListener("change", () => { state.shown = PAGE; renderCards(); });
$("#selTop").addEventListener("click", () => { state.picked = new Set(visibleCandidates().filter(c => c.score != null).slice(0, 3).map(c => c.id)); renderCards(); });
$("#selNone").addEventListener("click", () => { state.picked = new Set(); renderCards(); });
$("#render").addEventListener("click", async () => {
  const p = state.project; if (!p || !state.picked.size) return;
  const ids = [...state.picked].sort((a, b) => a - b), overrides = {};
  for (const id of ids) {
    const e = state.edits[id]; if (!e) continue;
    if (!(e.start >= 0 && e.end - e.start >= 3 && e.end - e.start <= 180)) { toast(`Clip #${id + 1}: use a start before the end, 3 to 180 seconds long.`, "bad"); return; }
    overrides[id] = e;
  }
  if (ids.length > 12) { toast("Render at most 12 clips at a time.", "bad"); return; }
  try { trackJob(await post(`/api/projects/${ytId()}/render`, { ids, overrides, ...renderOptions() })); updateRenderButton(); toast("Rendering started. The progress is shown at the top.", "good"); }
  catch (e) { toast(e.message, "bad"); }
});
$("#kitSel").addEventListener("click", async () => {
  if (!state.picked.size) return;
  try { trackJob(await post(`/api/projects/${ytId()}/kit`, { ids: [...state.picked].sort((a, b) => a - b) })); } catch (e) { toast(e.message, "bad"); }
});
$("#btnFolder").addEventListener("click", async () => { try { await post(`/api/projects/${ytId()}/open-folder`); } catch (e) { toast(e.message, "bad"); } });
$("#btnDelete").addEventListener("click", async () => {
  const p = state.project; if (!p || !confirm(`Delete “${p.title}” and all the clips rendered from it? This cannot be undone.`)) return;
  try { await api(`/api/projects/${encodeURIComponent(p.id)}`, { method: "DELETE" }); state.project = null; toast("Deleted."); go("projects"); }
  catch (e) { toast(e.message, "bad"); }
});
$("#btnRerun").addEventListener("click", () => $("#rerunBox").classList.toggle("hidden"));
$("#btnRerunGo").addEventListener("click", async () => {
  const p = state.project; if (!p) return;
  try {
    trackJob(await post(`/api/projects/${encodeURIComponent(p.id)}/reanalyze`, analyzeBody({ focus: $("#rFocus").value.trim(),
      min_seconds: +$("#rMin").value || 20, max_seconds: +$("#rMax").value || 59, refresh: false })));
    $("#rerunBox").classList.add("hidden");
  } catch (e) { toast(e.message, "bad"); }
});
$("#addOwn").addEventListener("click", () => { $("#acErr").textContent = ""; $("#acStart").value = $("#acEnd").value = $("#acTitle").value = ""; $("#addClip").showModal(); });
$("#acOk").addEventListener("click", async () => {
  const s = parseTime($("#acStart").value), e = parseTime($("#acEnd").value);
  if (Number.isNaN(s) || Number.isNaN(e) || s >= e) { $("#acErr").textContent = "Enter a start before the end, like 12:34 and 13:10."; return; }
  if (e - s < 3 || e - s > 59) { $("#acErr").textContent = "A Short stays under a minute: use 3 to 59 seconds."; return; }
  try {
    const cand = await post(`/api/projects/${ytId()}/candidates`, { start: s, end: e, title: $("#acTitle").value.trim() });
    $("#addClip").close(); await refreshProject(); state.picked.add(cand.id); state.shown = Math.max(state.shown, state.project.candidates.length); renderCards();
    const t = $("#card-" + cand.id); if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); t.classList.add("flash"); }
  } catch (err) { $("#acErr").textContent = err.message; }
});

/* ---------- YouTube page ---------- */
const safeAvatar = u => { try { const h = new URL(u).hostname; return /(^|\.)ggpht\.com$|(^|\.)googleusercontent\.com$/.test(h) ? u : null; } catch { return null; } };
async function loadYouTube() {
  try { state.yt = await api("/api/youtube"); } catch { /* keep what we had */ }
  return state.yt;
}
async function loadUploads() {
  try { const d = await api("/api/uploads"); state.uploads = d.items; state.yt = d.status; } catch { /* keep */ }
  renderUploads();
}
async function loadYouTubePage() {
  await loadYouTube(); renderYtPage(); loadUploads();
  try {                                                         // preferences
    const s = await api("/api/settings");
    $("#ytPrivacy").value = s.youtube_privacy; $("#ytCategory").value = s.youtube_category; $("#ytDaily").value = s.youtube_daily_uploads;
    $("#ytAuto").checked = !!s.youtube_auto_resume; $("#ytNotify").checked = !!s.youtube_notify_subscribers; $("#ytDelete").checked = s.delete_uploaded !== false;
    $("#ytId").value = s.youtube_client_id || "";
  } catch { /* the page still works */ }
  clearInterval(state.upTimer);
  state.upTimer = setInterval(async () => { if (currentIs("youtube")) { await loadYouTube(); renderYtPage(); loadUploads(); } }, 8000);
}
const stepsDone = () => { try { return JSON.parse(store.get("ytSteps") || "{}"); } catch { return {}; } };
function renderYtPage() {
  const y = state.yt || {}, card = $("#ytStatus");
  const avatar = y.channel && safeAvatar(y.channel.thumbnail);
  let main;
  if (y.connected) main = el("div", {}, el("div", { class: "big" }, el("span", { class: "ok", text: "● Connected" }), " to “" + ((y.channel && y.channel.title) || "your channel") + "”"),
    el("div", { class: "small muted", text: y.paused ? "Uploads are paused." : (y.parked_until ? "YouTube asked us to wait; uploads continue by themselves." : "Ready. Upload from any video, or use Autopilot on the Create page.") }));
  else if (y.needs_reconnect) main = el("div", {}, el("div", { class: "big" }, el("span", { class: "warn", text: "● Sign-in expired" })), el("div", { class: "small muted", text: (y.reconnect_reason || "") + " Press Reconnect. Queued clips wait until then." }));
  else main = el("div", {}, el("div", { class: "big" }, el("span", { class: "bad", text: "● Not connected" })), el("div", { class: "small muted", text: "Follow the five steps below once. After that, uploading is one click, or none with Autopilot." }));
  card.replaceChildren(...[avatar ? el("img", { src: avatar, alt: "", referrerpolicy: "no-referrer" }) : null, main].filter(Boolean));   // a null child would print "null"
  const meter = $("#ytMeterBox"); meter.classList.toggle("hidden", !y.connected);
  if (y.connected) {
    $("#ytMeter").style.width = Math.min(100, Math.round(100 * y.uploads_today / Math.max(1, y.daily_limit))) + "%";
    $("#ytMeterText").textContent = `${y.uploads_today} of ${y.daily_limit} uploads used today (YouTube's day ends at midnight Pacific time)`;
  }
  const pause = $("#ytPause"); pause.hidden = !y.connected; pause.textContent = y.paused ? "▶ Resume uploads" : "⏸ Pause uploads";
  $("#ytReconnect").hidden = !(y.connected || y.needs_reconnect); $("#ytChange").hidden = !y.connected; $("#ytDisconnect").hidden = !(y.connected || y.needs_reconnect);
  const showWizard = !y.connected || state.wizardOpen; $("#ytWizard").hidden = !showWizard;
  /* the steps: 1-3 happen in Google's console, so they are ticked by the person (or implied once the key exists) */
  const done = stepsDone(), impl = y.configured || y.connected, st = { 1: done[1] || impl, 2: done[2] || impl, 3: done[3] || impl, 4: !!y.configured, 5: !!y.connected };
  let now = 0;
  $$("#ytSteps li").forEach(li => {
    const n = +li.dataset.step, ok = !!st[n]; if (!ok && !now) now = n;
    li.classList.toggle("done", ok); li.classList.toggle("now", !ok && now === n);
    const b = $("button.done", li); if (b) { b.textContent = ok ? "✓ Done" : "I did this"; b.disabled = impl; }
  });
  const s4 = $("#ytStep4State"); s4.className = "autostate" + (y.configured ? " ok" : ""); s4.textContent = y.configured ? "✓ Key saved" : "";
  const s5 = $("#ytStep5State"); s5.className = "autostate" + (y.connected ? " ok" : ""); s5.textContent = y.connected ? "✓ Connected" : "";
  $("#ytConnect").textContent = y.connected || y.needs_reconnect ? "Reconnect" : "Connect YouTube"; $("#ytConnect").disabled = !y.configured;
  $("#ytSecretState").textContent = y.client_secret_set ? "(saved)" : "(not set)";
  if (!$("#ytId").value && y.client_id) $("#ytId").value = y.client_id;
  renderChips(); renderBanner();
}
$("#ytSteps").addEventListener("click", e => {
  const b = e.target.closest("button[data-done]"); if (!b) return;
  const d = stepsDone(), n = b.dataset.done; d[n] = !d[n]; store.set("ytSteps", JSON.stringify(d)); renderYtPage();
});
async function setPaused(paused) {
  try { state.yt = await post("/api/youtube/pause", { paused }); renderYtPage(); renderBanner(); toast(paused ? "Uploads paused." : "Uploads resumed.", "good"); loadUploads(); } catch (e) { toast(e.message, "bad"); }
}
$("#ytPause").addEventListener("click", () => setPaused(!(state.yt && state.yt.paused)));
$("#ytChange").addEventListener("click", () => { state.wizardOpen = !state.wizardOpen; renderYtPage(); if (state.wizardOpen) $("#ytWizard").scrollIntoView({ behavior: "smooth" }); });
$("#ytReconnect").addEventListener("click", connectYouTube);

/* the key file: dropped, chosen, or pasted */
async function saveYtClient(text) {
  const msg = $("#ytCheckMsg"); msg.className = "small muted"; msg.textContent = "Saving and checking the key with Google…";
  try {
    if (text) await post("/api/youtube/credentials", { json: text });
    else {
      const id = $("#ytId").value.trim(), secret = $("#ytSecret").value.trim();
      if (secret) await post("/api/youtube/credentials", { client_id: id, client_secret: secret });
      else if (id) await post("/api/settings", { youtube_client_id: id });
      else throw new Error("Drop the JSON file from Google, or type the client id and secret.");
    }
    $("#ytSecret").value = "";
    const r = await post("/api/youtube/check");
    msg.className = "small " + (r.ok ? "ok" : "err");
    msg.textContent = r.ok ? "✓ Saved. Google accepts this key. Now do step 5." : "✗ " + r.error;
    await loadConfig(); await loadYouTube(); renderYtPage();
  } catch (e) { msg.className = "small err"; msg.textContent = "✗ " + e.message; }
}
const drop = $("#ytDrop");
drop.addEventListener("click", () => $("#ytFile").click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#ytFile").click(); } });
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); });
async function readKeyFile(f) {
  if (!f) return;
  if (f.size > 50000) { toast("That file is too big to be a Google client_secret.json.", "bad"); return; }
  await saveYtClient(await f.text());
}
drop.addEventListener("drop", e => readKeyFile(e.dataTransfer.files[0]));
$("#ytFile").addEventListener("change", async e => { try { await readKeyFile(e.target.files[0]); } finally { e.target.value = ""; } });
drop.addEventListener("paste", e => { const t = e.clipboardData && e.clipboardData.getData("text"); if (t && t.includes("client_id")) { e.preventDefault(); saveYtClient(t.slice(0, 20000)); } });
$("#ytSaveClient").addEventListener("click", () => saveYtClient(null));

async function connectYouTube() {
  const win = window.open("", "_blank");                                    // opened inside the click, so the pop-up blocker lets it through
  try {
    const before = state.yt && state.yt.connected_at;
    const { url } = await post("/api/youtube/connect");
    if (win) win.location.href = url;
    else $("#ytCheckMsg").replaceChildren(el("a", { href: url, target: "_blank", rel: "noopener", text: "Pop-up blocked: click here to open the Google sign-in" }));
    toast("Sign in with Google in the new tab. Keep every permission box ticked.");
    clearInterval(state.ytPoll); const until = Date.now() + 5 * 60e3;
    state.ytPoll = setInterval(async () => {
      const y = await loadYouTube(); renderYtPage();
      if (y && y.connected && y.connected_at !== before) {
        clearInterval(state.ytPoll); state.wizardOpen = false; await loadConfig(); renderYtPage(); loadUploads();
        toast("YouTube connected" + (y.channel ? ": " + y.channel.title : "") + (y.testing_mode ? ". Publish your consent screen so it lasts (see the alert above)." : "."), y.testing_mode ? "" : "good");
      } else if (Date.now() > until) clearInterval(state.ytPoll);
    }, 2000);
  } catch (err) { if (win) win.close(); toast(err.message, "bad"); }
}
$("#ytConnect").addEventListener("click", connectYouTube);
$("#ytDisconnect").addEventListener("click", async () => {
  if (!confirm("Disconnect YouTube? Clips already on your channel stay there; waiting uploads will not go until you connect again.")) return;
  try { state.yt = await post("/api/youtube/disconnect"); state.wizardOpen = false; await loadConfig(); renderYtPage(); toast("YouTube disconnected.", "good"); } catch (err) { toast(err.message, "bad"); }
});
$("#ytSavePrefs").addEventListener("click", async () => {
  try {
    await post("/api/settings", { youtube_privacy: $("#ytPrivacy").value, youtube_category: $("#ytCategory").value,
      youtube_daily_uploads: String(Math.max(1, Math.min(50, Math.round(+$("#ytDaily").value) || 10))),
      youtube_auto_resume: $("#ytAuto").checked ? "1" : "0", youtube_notify_subscribers: $("#ytNotify").checked ? "1" : "0",
      delete_uploaded: $("#ytDelete").checked ? "1" : "0" });
    await loadConfig(); await loadYouTube(); renderYtPage(); toast("Upload settings saved.", "good");
  } catch (e) { toast(e.message, "bad"); }
});

/* the upload log: waiting / on YouTube / problems, across all videos */
function renderUploads() {
  const items = state.uploads || [];
  const waiting = items.filter(i => UP_BUSY.includes(i.state)), done = items.filter(i => i.state === "uploaded"), bad = items.filter(i => i.state === "failed");
  $("#ytCntWaiting").textContent = String(waiting.length || ""); $("#ytCntDone").textContent = String(done.length || ""); $("#ytCntProblems").textContent = String(bad.length || "");
  $$("#ytTabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === state.ytTab));
  $("#ytClearQueue").hidden = waiting.length === 0;
  const y = state.yt || {};
  $("#ytTryNow").hidden = !(waiting.length && y.connected && !y.paused && waiting.some(i => i.state === "queued"));
  const list = { waiting, done, problems: bad }[state.ytTab] || [];
  const openVideo = i => go("project/" + encodeURIComponent(i.project_id));
  const rows = list.map(i => {
    const acts = [];
    if (i.state === "uploaded") {
      acts.push(el("a", { href: i.url, target: "_blank", rel: "noopener", text: "▶ Watch" }), el("a", { href: i.studio_url, target: "_blank", rel: "noopener", text: "Studio" }));
    } else if (i.state === "queued") acts.push(el("button", { class: "small", onclick: async () => { try { await post(`/api/projects/${encodeURIComponent(i.project_id)}/upload/dequeue`, { files: [i.file] }); loadUploads(); loadConfig(); } catch (e) { toast(e.message, "bad"); } }, text: "Remove" }));
    else if (i.state === "failed") acts.push(el("button", { class: "small", onclick: async () => { try { const r = await post(`/api/projects/${encodeURIComponent(i.project_id)}/upload`, { files: [i.file], privacy: i.privacy || undefined }); if (r.job) trackJob(r.job); loadUploads(); } catch (e) { toast(e.message, "bad"); } }, text: "Try again" }));
    acts.push(el("button", { class: "small", onclick: () => openVideo(i), text: "Open video" }));
    const when = i.state === "uploaded" ? "uploaded " + localTime(i.uploaded) + (i.publish_at ? " · goes live " + localTime(i.publish_at) : " · " + (i.actual_privacy || i.privacy || ""))
      : i.state === "uploading" ? "uploading now…" : i.state === "queued" ? waitingWhy() + (i.publish_at ? " · goes live " + localTime(i.publish_at) : "") : "not uploaded";
    return el("div", { class: "yrow" },
      el("div", {}, el("b", { text: i.title || i.file }), el("span", { class: "small muted", text: `${i.project_title} · ${when}` })),
      el("div", { class: "acts" }, acts),
      i.error ? el("div", { class: "why " + (i.state === "failed" ? "err" : "muted"), text: i.error }) : null,
      i.notice ? el("div", { class: "why notice", style: "color:var(--warn)", text: i.notice }) : null);
  });
  const empty = { waiting: "Nothing is waiting. Clips you upload while YouTube is busy, or beyond your daily limit, show up here.",
                  done: "Nothing uploaded yet. Your first upload will appear here with its link.", problems: "No problems. 🎉" }[state.ytTab];
  $("#ytList").replaceChildren(...(rows.length ? rows : [el("div", { class: "empty", text: empty })]));
}
$("#ytTabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { state.ytTab = b.dataset.tab; renderUploads(); } });
$("#ytTryNow").addEventListener("click", tryUploadsNow);
$("#ytClearQueue").addEventListener("click", async () => {
  if (!confirm("Take every waiting upload out of the queue? Clips already on YouTube are not touched.")) return;
  try { const r = await post("/api/youtube/queue/clear"); toast(`${plural(r.removed, "clip", "clips")} removed from the queue.`, "good"); loadUploads(); loadConfig(); if (state.project) refreshProject(); } catch (e) { toast(e.message, "bad"); }
});

/* ---------- settings ---------- */
function syncProviderFields() {
  const gem = $("#sProvider").value === "gemini";
  $("#sModelWrap").classList.toggle("hidden", !gem); $("#sGKeyWrap").classList.toggle("hidden", !gem);
  $("#sOKeyWrap").classList.toggle("hidden", gem); $("#sOModelWrap").classList.toggle("hidden", gem);
}
$("#sProvider").addEventListener("change", syncProviderFields);
function settingsPane(name) {
  $$("#sTabs button").forEach(b => b.classList.toggle("on", b.dataset.pane === name));
  $$(".spane").forEach(p => p.classList.toggle("hidden", p.dataset.pane !== name));
  if (name === "health") loadHealth(false);
  if (name === "disk") loadUsage().catch(() => {});
}
$("#sTabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) settingsPane(b.dataset.pane); });
async function loadUsage() {
  const u = await api("/api/storage");
  $("#sUsage").replaceChildren(...[["Rendered clips", u.clips], ["Saved transcripts and signals", u.cache], ["Saved AI answers", u.llm_cache], ["Old whole-video downloads", u.legacy_downloads]]
    .flatMap(([k, v]) => [el("span", { text: k }), el("span", { text: fmtBytes(v) })]));
  $("#cleanLegacy").hidden = !u.legacy_downloads;
  if (u.legacy_downloads) $("#cleanLegacy").textContent = `Delete old whole-video downloads (${fmtBytes(u.legacy_downloads)})`;
}
async function openSettings(pane = "ai") {
  const dlg = $("#settings"); if (!dlg.open) dlg.showModal();
  $("#sTest").textContent = ""; settingsPane(pane);
  try {
    const s = await api("/api/settings");
    $("#sProvider").value = s.llm_provider; $("#sOModel").value = s.openai_model; $("#sCookies").value = s.ytdlp_cookies_browser || "";
    $("#sWhisper").value = s.whisper_model; $("#sCookiesFile").value = s.ytdlp_cookies_file || ""; $("#sProxy").value = "";
    $("#sEncoder").value = s.encoder === "cpu" ? "cpu" : "auto";
    $("#sCacheDays").value = s.cache_days || 3;
    try { const a = await api("/api/app"); $("#sAutostart").checked = !!a.autostart; $("#sAutostart").disabled = !a.supported; } catch {}
    $("#sProxyState").replaceChildren(s.ytdlp_proxy_set
      ? el("span", {}, "(saved) ", el("button", { class: "small", type: "button", text: "remove", onclick: removeProxy }))
      : document.createTextNode("(none)"));
    $("#sGKeyState").textContent = s.gemini_key_set ? "(saved)" : "(not set)"; $("#sOKeyState").textContent = s.openai_key_set ? "(saved)" : "(not set)";
    $("#sGKey").value = $("#sOKey").value = ""; syncProviderFields();
    const sel = $("#sGeminiModel"); sel.replaceChildren(el("option", { value: "auto", text: "Auto: best available, falls back by itself" }));
    if (s.gemini_key_set) { try { (await api("/api/models")).models.forEach(m => sel.append(el("option", { value: m, text: m }))); } catch {} }
    if (![...sel.options].some(o => o.value === s.gemini_model)) sel.append(el("option", { value: s.gemini_model, text: s.gemini_model }));
    sel.value = s.gemini_model;
  } catch (e) { toast(e.message, "bad"); }
}
function settingsBody() {
  const b = { llm_provider: $("#sProvider").value, gemini_model: $("#sGeminiModel").value, openai_model: $("#sOModel").value.trim(),
              ytdlp_cookies_browser: $("#sCookies").value, ytdlp_cookies_file: $("#sCookiesFile").value.trim(), whisper_model: $("#sWhisper").value,
              encoder: $("#sEncoder").value, cache_days: String(Math.min(90, Math.max(1, +$("#sCacheDays").value || 3))) };
  if ($("#sProxy").value.trim()) b.ytdlp_proxy = $("#sProxy").value.trim();
  if ($("#sGKey").value.trim()) b.gemini_api_key = $("#sGKey").value.trim();
  if ($("#sOKey").value.trim()) b.openai_api_key = $("#sOKey").value.trim();
  return b;
}
async function testConnection() {
  const out = $("#sTest"); out.className = "small muted"; out.textContent = "Testing the connection…";
  const r = await post("/api/settings/test");
  out.className = "small " + (r.ok ? "" : "err");
  out.textContent = r.ok ? `✓ Working: ${r.provider} answered in ${r.seconds}s${r.model ? " (" + r.model + ")" : ""}.` : "✗ " + r.error;
  return r;
}
$("#openSettings").addEventListener("click", () => openSettings("ai"));
$("#sAutostart").addEventListener("change", async () => {
  try { await post("/api/app/autostart", { on: $("#sAutostart").checked }); toast($("#sAutostart").checked ? "It will start with Windows." : "It will not start by itself any more.", "good"); }
  catch (err) { toast(err.message, "bad"); }
});
$("#sQuit").addEventListener("click", async () => {
  if (!confirm("Quit Shorts Studio? Waiting uploads continue the next time it runs.")) return;
  try {
    await post("/api/quit");
    document.body.replaceChildren(el("p", { style: "padding:40px;font-size:18px", text: "Shorts Studio is closed. Open it again with its desktop icon." }));
  } catch (err) { toast(err.message, "bad"); }
});
$("#sClose").addEventListener("click", () => $("#settings").close());
$("#sSave").addEventListener("click", async () => {
  const typedKey = !!($("#sGKey").value.trim() || $("#sOKey").value.trim());
  try {
    await post("/api/settings", settingsBody()); await loadConfig(); toast("Settings saved.", "good");
    if (typedKey) { settingsPane("ai"); await openSettingsKeep(); await testConnection(); }          // a new key: prove that it works
    else $("#settings").close();
  } catch (e) { $("#sTest").className = "small err"; $("#sTest").textContent = "✗ " + e.message; }
});
async function openSettingsKeep() { const t = $("#sTest").textContent; await openSettings("ai"); $("#sTest").textContent = t; }
$("#sTestBtn").addEventListener("click", async () => { try { await testConnection(); } catch (e) { $("#sTest").textContent = "✗ " + e.message; } });
async function removeProxy() {
  try { await post("/api/settings", { ytdlp_proxy: "" }); toast("Proxy removed.", "good"); await openSettings("login"); } catch (e) { toast(e.message, "bad"); }
}
async function loadHealth(deep) {
  const box = $("#sHealth"); box.className = "small muted"; box.textContent = deep ? "Checking (this tests the network too)…" : "Checking…";
  try {
    const r = await api("/api/health" + (deep ? "?deep=1" : ""));
    const mark = { ok: "✓", warn: "!", fail: "✗" }, color = { ok: "var(--good)", warn: "var(--warn)", fail: "var(--bad)" };
    box.className = "small";
    box.replaceChildren(...r.checks.flatMap(c => [
      el("div", { style: "display:flex;gap:8px;margin:2px 0" },
        el("b", { style: `color:${color[c.status]};width:14px`, text: mark[c.status] }),
        el("span", {}, el("b", { text: c.label + ": " }), c.detail)),
      c.fix && c.status !== "ok" ? el("div", { class: "muted", style: "margin:0 0 4px 22px", text: "→ " + c.fix }) : null]).filter(Boolean));
    return r;
  } catch (e) { box.className = "small err"; box.textContent = "Could not run the health check: " + e.message; }
}
$("#hRecheck").addEventListener("click", () => loadHealth(true));
$("#hUpdate").addEventListener("click", async () => {
  try { trackJob(await post("/api/ytdlp/update")); toast("Updating yt-dlp… the progress is shown on the page behind this window."); }
  catch (e) { toast(e.message, "bad"); }
});
async function clean(what, label) {
  try { const r = await post("/api/storage/clean", { what }); toast(`${label}: freed ${fmtBytes(Object.values(r.freed).reduce((a, b) => a + b, 0))}.`, "good"); await loadUsage(); }
  catch (e) { toast(e.message, "bad"); }
}
$("#cleanLegacy").addEventListener("click", () => confirm("Delete the old whole-video downloads? They are not needed any more.") && clean(["legacy_downloads"], "Old downloads"));
$("#cleanCache").addEventListener("click", () => clean(["cache"], "Transcripts"));
$("#cleanLlm").addEventListener("click", () => clean(["llm_cache"], "AI answers"));

/* ---------- start ---------- */
(async function init() {
  try { JSON.parse(store.get("dismissed") || "[]").forEach(id => state.dismissed.add(id)); } catch { /* ignore */ }
  const legacy = new URLSearchParams(location.search).get("project");      // older links: /?project=<id>
  if (legacy && !location.hash) history.replaceState(null, "", location.pathname + "#/project/" + encodeURIComponent(legacy));
  try { await loadLibrary(); } catch { /* the first call below reports it */ }
  try { await loadConfig(); } catch (e) { toast("Cannot reach the app: " + e.message, "bad"); }
  await route();
  try {          /* a job keeps running on the server when the page is refreshed: pick it back up */
    const jobs = await api("/api/jobs"), recent = Date.now() / 1000 - 12 * 3600;
    for (const j of jobs.filter(x => ["queued", "running"].includes(x.status))) trackJob(await api("/api/jobs/" + j.id));
    /* work that was cut off when the app was closed: show it, with a Try again button */
    for (const j of jobs.filter(x => x.status === "interrupted" && x.created > recent && !state.dismissed.has(x.id))) state.jobs.set(j.id, await api("/api/jobs/" + j.id));
    renderJobs();
  } catch { /* ignore */ }
})();
