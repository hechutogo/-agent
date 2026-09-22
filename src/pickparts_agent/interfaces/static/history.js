"use strict";
const $ = id => document.getElementById(id);
const U = TraceUI;
const listRequest = new U.LatestRequest();
const detailRequest = new U.LatestRequest();
const lanes = new WeakMap();
let runs = [], selected = null, detail = null, scope = null;
let listError = "", detailError = "", listLoaded = false, polling = false;
let logFollowing = true, logSignature = "", pollTimer;

function node(tag, className = "", text) {
  const result = document.createElement(tag);
  result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function text(id, value) {
  const target = $(id);
  const next = String(value ?? "");
  if (target.textContent !== next) target.textContent = next;
}

// Keep live rows in place so polling does not close disclosures or steal focus.
function syncRows(lane, entries, keyFor, create, update) {
  const previous = lanes.get(lane) || new Map(), next = new Map();
  entries.forEach((entry, index) => {
    const key = keyFor(entry, index);
    const view = previous.get(key) || create(entry, index);
    update(view, entry, index);
    if (lane.children[index] !== view.node) lane.insertBefore(view.node, lane.children[index] || null);
    next.set(key, view);
  });
  previous.forEach((view, key) => { if (!next.has(key)) view.node.remove(); });
  lanes.set(lane, next);
}

function localDay(date = new Date()) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function timeLabel(value, full = false) {
  const time = U.timestamp(value);
  if (time === null) return "时间未记录";
  return new Date(time).toLocaleString("zh-CN", full
    ? { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }
    : { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function filters() {
  return { agent: $("run-agent").value, status: $("run-status").value, text: $("run-search").value };
}

function writeURL(push = false) {
  const url = new URL(location.href);
  const values = { run: selected, day: $("run-day").value, ...filters() };
  Object.entries(values).forEach(([key, value]) => value ? url.searchParams.set(key, value) : url.searchParams.delete(key));
  if (url.href !== location.href) history[push ? "pushState" : "replaceState"](null, "", url);
}

function connection() {
  const error = listError || detailError;
  text("history-connection", error ? "连接异常 · 自动重试" : "实时同步 · 每 3 秒更新");
  $("history-dot").classList.toggle("offline", Boolean(error));
}

async function getJSON(path, ticket) {
  const response = await fetch(path, {
    cache: "no-store", signal: AbortSignal.any([ticket.signal, AbortSignal.timeout(12000)]),
  });
  if (!response.ok) throw Error(response.status === 404
    ? "该运行不存在或已超出日志保留期。" : `读取失败（HTTP ${response.status}），正在自动重试。`);
  return response.json();
}

function renderList() {
  const visible = U.filterRuns(runs, filters());
  text("run-count", `${visible.length} / ${runs.length}`);
  $("list-empty").hidden = visible.length > 0;
  if (!visible.length && listLoaded) {
    $("list-empty").replaceChildren(node("span", "empty-glyph", "⌁"),
      node("strong", "", runs.length ? "没有匹配的运行" : "这一天暂无运行记录"),
      node("span", "", runs.length ? "尝试调整 Agent、状态或搜索词。" : "选择其他日期，或返回控制台开始一次任务。"));
  }
  const outside = selected && listLoaded && !visible.some(run => run.run_id === selected);
  $("list-notice").hidden = !listError && !outside;
  text("list-notice", listError || "当前详情不在此筛选结果中，仍会保持选中并更新。");
  syncRows($("run-list"), visible, run => run.run_id, run => {
    const wrapper = node("div", "run-list-entry");
    wrapper.setAttribute("role", "listitem");
    const button = node("button", "run-item");
    button.type = "button";
    const heading = node("span", "run-item-heading"), agent = node("span", "run-agent-name");
    const status = node("span", "status-badge"), title = node("span", "run-task");
    const goal = node("span", "run-goal"), meta = node("span", "run-meta"), id = node("code", "run-id");
    heading.append(agent, status);
    button.append(heading, title, goal, meta, id);
    button.addEventListener("click", () => selectRun(run.run_id, true));
    wrapper.append(button);
    return { node: wrapper, button, agent, status, title, goal, meta, id };
  }, (view, run) => {
    const status = U.runStatus(run);
    view.button.classList.toggle("active", run.run_id === selected);
    view.button.setAttribute("aria-current", run.run_id === selected ? "true" : "false");
    view.agent.textContent = U.agentName(run.agent);
    view.status.className = `status-badge ${U.statusTone(status)}`;
    view.status.textContent = U.statusName(status);
    view.title.textContent = U.taskName(run.task);
    view.goal.textContent = run.goal || run.message || "未记录任务描述";
    view.id.textContent = run.run_id;
    view.meta.textContent = `${timeLabel(run.start_ts)} · ${U.durationLabel(U.timeline(run).duration)}`;
  });
}

function reconcileRun(current, incoming) {
  if (!current || current.run_id !== incoming.run_id) return incoming;
  const currentTerminal = U.runStatus(current) !== "running";
  const incomingTerminal = U.runStatus(incoming) !== "running";
  const newer = currentTerminal !== incomingTerminal ? incomingTerminal
    : (U.timestamp(incoming.end_ts) ?? incoming.duration_ms ?? 0)
      >= (U.timestamp(current.end_ts) ?? current.duration_ms ?? 0);
  return newer ? { ...current, ...incoming } : { ...incoming, ...current };
}

async function loadRuns() {
  const ticket = listRequest.begin();
  const day = $("run-day").value || localDay();
  $("run-list").setAttribute("aria-busy", "true");
  try {
    const data = await getJSON(`/api/runs?day=${encodeURIComponent(day)}&limit=500`, ticket);
    if (!ticket.current()) return;
    const previous = new Map(runs.map(run => [run.run_id, run]));
    runs = (Array.isArray(data.runs) ? data.runs : []).map(run => {
      const latest = reconcileRun(previous.get(run.run_id), run);
      return detail?.run_id === run.run_id ? reconcileRun(latest, detail) : latest;
    });
    listError = "";
    listLoaded = true;
    renderList();
    const first = U.filterRuns(runs, filters())[0];
    if (!selected && first) selectRun(first.run_id);
  } catch (error) {
    if (!ticket.current()) return;
    listError = error.name === "TimeoutError" ? "读取超时，正在自动重试。" : error.message || "无法连接服务，正在自动重试。";
    renderList();
    if (!listLoaded) $("list-empty").replaceChildren(node("strong", "", "暂时无法读取记录"), node("span", "", "点击刷新重试，或等待自动重连。"));
  } finally {
    if (ticket.current()) {
      $("run-list").setAttribute("aria-busy", "false");
      connection();
    }
  }
}

function spanLabel(span) {
  const names = {
    plan: "理解与规划", execute: "执行计划", react: "ReAct 纠偏",
    tiptop: "TiPToP 开环任务", ground: "指令理解与目标接地", perceive: "本地 RGB-D 感知",
    "vision:locate": "视觉定位", "tiptop:perception": "本地 RGB-D 感知",
    "tiptop:plan": "TAMP 一次性规划", "tiptop:execute": "开环执行",
  };
  const atoms = { find_object: "定位物体", reach_above: "移动到上方", grasp: "抓取", lift: "抬起",
    carry_to: "搬运", place_on: "叠放", release_into: "放入容器", place_on_table: "放到桌面",
    verify_state: "视觉校验", reset_arm: "机械臂复位", set_gripper: "设置夹爪",
    move: "移动", gripper: "夹爪", calibrate: "固定抓取标定", reset: "机械臂复位" };
  if (detail?.agent === "tiptop" && span.name === "plan") return "TAMP 一次性规划";
  if (detail?.agent === "tiptop" && span.name === "execute") return "开环执行计划";
  if (names[span.name]) return names[span.name];
  if (span.name?.startsWith("run:")) return U.taskName(span.name.slice(4));
  if (span.name?.startsWith("atom:")) return atoms[span.name.slice(5)] || `动作 · ${span.name.slice(5)}`;
  if (span.name?.startsWith("llm:")) return `模型调用 · ${span.name.slice(4)}`;
  return span.name || "未命名调用";
}

function scopeIds() {
  if (!scope || !detail) return null;
  const ids = new Set([scope]);
  const children = new Map();
  (detail.spans || []).forEach(span => {
    if (!children.has(span.parent_id)) children.set(span.parent_id, []);
    children.get(span.parent_id).push(span.span_id);
  });
  const queue = [scope];
  for (let index = 0; index < queue.length; index++) {
    (children.get(queue[index]) || []).forEach(id => { if (!ids.has(id)) { ids.add(id); queue.push(id); } });
  }
  return ids;
}

function setScope(id) {
  scope = id;
  renderEvidence();
  renderSpans();
}

function artifactLink(artifact, runId) {
  const href = U.artifactURL(runId, artifact.path);
  const link = node(href ? "a" : "span", "artifact-link",
    `${({ rgbd: "RGB-D 观测", state: "场景状态", llm: "模型输入输出" })[artifact.type] || artifact.type || "产物"} · ${artifact.note || artifact.path?.split("/").pop() || "未命名"}`);
  if (href) { link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer"; }
  else link.title = "产物路径无效或不属于当前运行";
  return link;
}

function renderSpans() {
  if (!detail) return;
  const spans = detail.spans || [], axis = U.timeline(detail);
  text("span-count", spans.length);
  text("axis-middle", U.durationLabel(axis.duration / 2));
  text("axis-end", U.durationLabel(axis.duration));
  $("spans-empty").hidden = spans.length > 0;
  const artifactsBySpan = new Map(), logCounts = new Map();
  (detail.artifacts || []).forEach(artifact => {
    if (!artifactsBySpan.has(artifact.span_id)) artifactsBySpan.set(artifact.span_id, []);
    artifactsBySpan.get(artifact.span_id).push(artifact);
  });
  (detail.logs || []).forEach(log => logCounts.set(log.span_id, (logCounts.get(log.span_id) || 0) + 1));
  syncRows($("span-rows"), U.orderedSpans(spans), row => row.span.span_id, ({ span }) => {
    const disclosure = node("details", "trace-span"), summary = node("summary", "span-summary");
    const identity = node("span", "span-identity"), title = node("span", "span-label");
    const status = node("span", "span-state"), track = node("span", "span-track"), bar = node("span", "span-bar");
    const duration = node("span", "span-duration"), content = node("div", "span-detail");
    const code = node("code", "span-technical"), attrs = node("pre", "span-properties"), error = node("p", "span-error");
    const links = node("div", "span-artifacts"), action = node("button", "text-button");
    action.type = "button";
    action.addEventListener("click", () => setScope(scope === span.span_id ? null : span.span_id));
    identity.append(title, status); track.append(bar); summary.append(identity, track, duration);
    content.append(code, error, attrs, action, links); disclosure.append(summary, content);
    return { node: disclosure, identity, title, status, track, bar, duration, code, attrs, error, links, action, artifactSignature: "" };
  }, (view, { span, depth }) => {
    const status = U.runStatus(span), timing = U.spanTiming(span, axis);
    view.node.className = `trace-span ${U.statusTone(status)}${scope === span.span_id ? " scoped" : ""}`;
    view.identity.style.paddingLeft = `${Math.min(depth, 12) * 14 + 14}px`;
    view.title.textContent = spanLabel(span);
    view.title.title = `${span.name} · 层级 ${depth + 1}`;
    view.status.textContent = U.statusName(status);
    view.bar.style.left = `${timing.offset}%`;
    view.bar.style.width = `${timing.width}%`;
    view.bar.hidden = !timing.known;
    view.track.title = timing.known ? `起点 +${U.durationLabel(U.timestamp(span.start_ts) - axis.start)} · ${U.durationLabel(timing.duration)}` : "未记录起始时间";
    view.duration.textContent = timing.known || span.duration_ms != null ? U.durationLabel(timing.duration) : "—";
    view.code.textContent = `${span.name || ""} · ${span.span_id}\n${timeLabel(span.start_ts, true)} → ${span.end_ts ? timeLabel(span.end_ts, true) : status === "running" ? "执行中" : "结束时间未记录"}${span.parent_id ? `\n父调用：${span.parent_id}` : ""}`;
    const properties = JSON.stringify(span.attrs || {}, null, 2);
    if (view.attrs.textContent !== properties) view.attrs.textContent = properties;
    view.attrs.hidden = !Object.keys(span.attrs || {}).length;
    view.error.textContent = [span.error_kind, span.error_message].filter(Boolean).join(" · ");
    view.error.hidden = !view.error.textContent;
    view.action.textContent = scope === span.span_id ? "取消调用筛选" : `关联日志与产物（本层 ${logCounts.get(span.span_id) || 0} 条日志，含子调用）`;
    view.action.setAttribute("aria-pressed", String(scope === span.span_id));
    const artifacts = artifactsBySpan.get(span.span_id) || [], signature = JSON.stringify(artifacts);
    if (signature !== view.artifactSignature) {
      view.links.replaceChildren(...artifacts.map(artifact => artifactLink(artifact, detail.run_id)));
      view.artifactSignature = signature;
    }
  });
}

function updateFollow() {
  $("log-follow").checked = logFollowing;
  $("log-latest").hidden = logFollowing || $("log-lane").scrollHeight <= $("log-lane").clientHeight + 24;
}

function latestLogs() {
  logFollowing = true;
  $("log-lane").scrollTop = $("log-lane").scrollHeight;
  updateFollow();
}

function renderEvidence() {
  if (!detail) return;
  const ids = scopeIds(), allLogs = detail.logs || [];
  const logs = U.filterLogs(allLogs, { spanIds: ids, level: $("log-level").value, text: $("log-search").value });
  $("span-scope").hidden = !scope;
  const scoped = (detail.spans || []).find(span => span.span_id === scope);
  text("span-scope-label", scoped ? `关联调用：${spanLabel(scoped)}（含子调用）` : "关联调用");
  text("log-count", `${logs.length} / ${allLogs.length}`);
  $("logs-empty").hidden = logs.length > 0;
  text("logs-empty", allLogs.length ? "没有匹配的日志，尝试调整级别、搜索词或调用筛选。" : "暂无日志，运行中的记录会自动出现。");
  syncRows($("log-rows"), logs, (log, index) => log.seq ?? `${log.ts}:${index}`, log => {
    const row = node("div", "trace-log"), time = node("span", "log-time"), level = node("span", "log-level");
    const message = node("span", "log-message"), source = node("button", "log-source");
    source.type = "button";
    source.addEventListener("click", () => {
      const view = lanes.get($("span-rows"))?.get(log.span_id);
      if (!view) return;
      view.node.open = true;
      const pane = $("span-pane"), rect = view.node.getBoundingClientRect(), bounds = pane.getBoundingClientRect();
      pane.scrollTop += rect.top - bounds.top - 40;
      view.node.querySelector("summary").focus({ preventScroll: true });
    });
    row.append(time, level, message, source);
    return { node: row, time, level, message, source };
  }, (view, log) => {
    const level = String(log.level || "info").toLowerCase();
    view.node.className = `trace-log ${["error", "critical", "warning", "debug"].includes(level) ? level : "info"}`;
    view.time.textContent = timeLabel(log.ts);
    view.time.title = `${log.ts || ""}${log.seq != null ? ` · #${log.seq}` : ""}`;
    view.level.textContent = level.toUpperCase();
    view.message.textContent = [log.message, log.exc_type].filter(Boolean).join(" · ");
    view.source.textContent = log.logger || (log.span_id ? "关联调用 ↗" : "");
    view.source.disabled = !(detail.spans || []).some(span => span.span_id === log.span_id);
    view.source.title = log.span_id || "未关联调用";
  });
  const signature = JSON.stringify([selected, scope, $("log-level").value, $("log-search").value, logs.length, logs.at(-1)]);
  if (signature !== logSignature && logFollowing) latestLogs();
  logSignature = signature;
  updateFollow();
  const artifacts = (detail.artifacts || []).filter(artifact => !ids || ids.has(artifact.span_id));
  text("artifact-count", artifacts.length);
  $("artifacts-empty").hidden = artifacts.length > 0;
  text("artifacts-empty", scope ? "此调用及其子调用暂无产物。" : "此次运行暂无产物。");
  syncRows($("artifact-list"), artifacts, (artifact, index) => `${artifact.path}:${index}`, artifact => {
    const card = node("div", "artifact-card"), link = artifactLink(artifact, detail.run_id);
    const path = node("code", "artifact-path"), source = node("button", "artifact-source");
    source.type = "button";
    source.addEventListener("click", () => setScope(artifact.span_id));
    card.append(link, path, source);
    return { node: card, path, source };
  }, (view, artifact) => {
    view.path.textContent = artifact.path?.split("/").slice(2).join("/") || "路径未记录";
    const span = (detail.spans || []).find(item => item.span_id === artifact.span_id);
    view.source.hidden = !span;
    view.source.textContent = span ? `关联 ${spanLabel(span)}` : "";
  });
}

function renderDetail() {
  if (!detail) return;
  const status = U.runStatus(detail), counts = U.runCounts(detail), axis = U.timeline(detail);
  $("detail-empty").hidden = true;
  $("detail-content").hidden = false;
  text("run-title", U.taskName(detail.task));
  text("run-agent-label", U.agentName(detail.agent));
  $("run-state").hidden = false;
  $("run-state").className = `status-badge ${U.statusTone(status)}`;
  text("run-state", U.statusName(status));
  text("detail-run-id", detail.run_id);
  text("run-time", timeLabel(detail.start_ts, true));
  $("run-permalink").href = `/history?run=${encodeURIComponent(detail.run_id)}&day=${encodeURIComponent($("run-day").value)}`;
  text("run-goal-text", detail.goal || "本次为场景或引擎操作，未记录自然语言指令。");
  text("metric-duration", U.durationLabel(axis.duration));
  text("metric-atoms", counts.atoms);
  text("metric-llm", counts.llm_calls);
  text("metric-react", counts.react_iters);
  $("run-result").hidden = !detail.message && !detail.recovery_required;
  text("run-result", `${detail.message || ""}${detail.recovery_required ? " 需重置场景后才能继续。" : ""}`);
  $("run-result").className = `result-note ${U.statusTone(status)}`;
  $("run-verification").hidden = !["tiptop", "tiptop_optimized"].includes(detail.agent);
  text("run-verification", detail.agent === "tiptop_optimized"
    ? "TiPToP 优化版逐个执行子任务，通过视觉校验与 ReAct 纠偏确认结果；请结合运行结果与日志查看完成情况。"
    : "TiPToP 初版为一次性规划与开环执行。运行完成仅表示执行结束，不代表目标已通过视觉验证。");
  renderSpans();
  renderEvidence();
  text("detail-updated", `${status === "running" ? "运行中 · " : ""}更新于 ${timeLabel(new Date().toISOString())}`);
}

async function refreshDetail() {
  if (!selected) return;
  const runId = selected, ticket = detailRequest.begin();
  $("span-pane").setAttribute("aria-busy", "true");
  try {
    const data = await getJSON(`/api/runs/${encodeURIComponent(runId)}`, ticket);
    if (!ticket.current() || runId !== selected) return;
    if (data.run_id !== runId) throw Error("运行详情与请求不匹配，正在重新读取。");
    detail = reconcileRun(detail, data);
    detailError = "";
    $("detail-notice").hidden = true;
    renderDetail();
    const index = runs.findIndex(run => run.run_id === runId);
    if (index >= 0) {
      runs[index] = reconcileRun(runs[index], detail);
      renderList();
    }
  } catch (error) {
    if (!ticket.current() || runId !== selected) return;
    detailError = error.name === "TimeoutError" ? "详情读取超时，正在自动重试。" : error.message || "无法读取运行详情。";
    $("detail-notice").hidden = false;
    text("detail-notice", `${detailError}${detail ? " 当前保留上次成功读取的内容。" : ""}`);
    if (!detail) $("detail-empty").replaceChildren(node("span", "empty-glyph", "!"),
      node("strong", "", "暂时无法显示此运行"), node("span", "", "可点击刷新重试，或选择其他记录。"));
  } finally {
    if (ticket.current()) {
      $("span-pane").setAttribute("aria-busy", "false");
      connection();
    }
  }
}

function selectRun(runId, push = false, updateURL = true) {
  if (selected === runId && detail) return refreshDetail();
  detailRequest.cancel();
  selected = runId;
  detail = null;
  scope = null;
  detailError = "";
  logSignature = "";
  logFollowing = true;
  $("detail-content").hidden = true;
  $("detail-notice").hidden = true;
  $("run-state").hidden = true;
  $("detail-empty").hidden = false;
  text("run-title", runId ? "读取运行详情…" : "选择一次运行");
  text("run-agent-label", "RUN DETAIL");
  $("detail-empty").replaceChildren(
    node("span", runId ? "loader" : "empty-glyph", runId ? "" : "⌁"),
    node("strong", "", runId ? "正在读取完整追踪" : "从左侧选择一次运行"),
    node("span", "", runId || "查看动作、模型调用、日志与产物。"));
  ["span-rows", "log-rows", "artifact-list"].forEach(id => { $(id).replaceChildren(); lanes.delete($(id)); });
  $("detail-content").scrollTop = $("span-pane").scrollTop = $("log-lane").scrollTop = 0;
  updateFollow();
  if (updateURL) writeURL(push);
  renderList();
  if (runId) return refreshDetail();
}

function restoreURL() {
  const params = new URL(location.href).searchParams;
  const id = params.get("run"), dayFromId = /^(\d{4})(\d{2})(\d{2})-/.exec(id || "");
  $("run-day").value = params.get("day") || (dayFromId ? `${dayFromId[1]}-${dayFromId[2]}-${dayFromId[3]}` : localDay());
  if (!$("run-day").value) $("run-day").value = localDay();
  ["agent", "status"].forEach(key => { $(`run-${key}`).value = params.get(key) || ""; });
  $("run-search").value = params.get("text") || "";
  selectRun(id, false, false);
  loadRuns();
}

async function pollHistory() {
  if (polling) return;
  polling = true;
  clearTimeout(pollTimer);
  try {
    if (!document.hidden) await Promise.all([loadRuns(), refreshDetail()]);
  } finally {
    polling = false;
    pollTimer = setTimeout(pollHistory, 3000);
  }
}

$("run-day").addEventListener("change", () => {
  if (!$("run-day").value) $("run-day").value = localDay();
  runs = [];
  listLoaded = false;
  listError = "";
  renderList();
  $("list-empty").replaceChildren(node("span", "loader"), node("strong", "", "正在读取所选日期"));
  writeURL();
  loadRuns();
});
["run-agent", "run-status", "run-search"].forEach(id => $(id).addEventListener(id === "run-search" ? "input" : "change", () => {
  renderList();
  writeURL();
}));
$("clear-filters").addEventListener("click", () => {
  $("run-agent").value = $("run-status").value = $("run-search").value = "";
  renderList();
  writeURL();
});
["log-level", "log-search"].forEach(id => $(id).addEventListener(id === "log-search" ? "input" : "change", renderEvidence));
$("clear-span").addEventListener("click", () => setScope(null));
$("log-lane").addEventListener("scroll", () => {
  const lane = $("log-lane");
  if (lane.scrollHeight - lane.clientHeight - lane.scrollTop > 28) logFollowing = false;
  updateFollow();
}, { passive: true });
$("log-follow").addEventListener("change", () => {
  logFollowing = $("log-follow").checked;
  if (logFollowing) latestLogs();
  else updateFollow();
});
$("log-latest").addEventListener("click", latestLogs);
$("refresh-history").addEventListener("click", pollHistory);
window.addEventListener("popstate", restoreURL);
document.addEventListener("visibilitychange", () => { if (!document.hidden) pollHistory(); });
restoreURL();
pollTimer = setTimeout(pollHistory, 3000);
