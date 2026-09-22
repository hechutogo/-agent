/* Shared, dependency-free presentation rules for the console and trace viewer. */
(function (root) {
  "use strict";
  const agentName = id => ({
    pickparts: "自研 PickParts", tiptop: "TiPToP 初版", tiptop_optimized: "TiPToP 优化版",
  }[id] || id || "未记录 Agent");
  const taskName = task => ({
    command: "执行指令", reset: "重置场景", init: "初始化场景", startup: "初始化场景",
    add_object: "添加物体", object: "添加物体", add: "添加物体", switch: "切换 Agent", switch_agent: "切换 Agent", agent: "切换 Agent",
  }[task] || task || "运行记录");
  const runStatus = run => {
    const status = String(run.status || "running").toLowerCase();
    return ({ ok: "success", done: "success", completed: "success", failed: "error", failure: "error", open: "running" })[status] || status;
  };
  const statusName = status => ({
    success: "已完成", running: "运行中", incomplete: "未完成", error: "异常", aborted: "已中止",
    cancelled: "已取消", interrupted: "已中断", skipped: "已跳过",
  }[status] || status || "未知");
  const statusTone = status => status === "incomplete" ? "error" : ["success", "running", "error"].includes(status) ? status : "neutral";
  function completionLabel(message) {
    if (!message.success) return "未完成 · 请查看结果";
    if (message.agent === "tiptop" || message.verification === "open_loop") return "开环执行完成 · 未经视觉验证";
    return message.verification === "visual" ? "视觉确认完成" : "执行完成";
  }
  function filterRuns(runs, { agent = "", status = "", text = "" } = {}) {
    const needle = text.trim().toLowerCase();
    return runs.filter(run => (!agent || (run.agent || "unknown") === agent)
      && (!status || runStatus(run) === status)
      && (!needle || [run.run_id, run.task, taskName(run.task), run.goal, run.message, agentName(run.agent)]
        .join(" ").toLowerCase().includes(needle)));
  }
  function timestamp(value) {
    if (value == null || value === "") return null;
    const time = typeof value === "number" ? value : Date.parse(value);
    return Number.isFinite(time) ? time : null;
  }
  const durationValue = value => typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  function timeline(run, now = Date.now()) {
    const spans = run.spans || [];
    const starts = [timestamp(run.start_ts), ...spans.map(span => timestamp(span.start_ts))].filter(time => time !== null);
    const start = starts.length ? Math.min(...starts) : now;
    const ends = [start, timestamp(run.end_ts)];
    const duration = durationValue(run.duration_ms);
    if (duration !== null) ends.push(start + duration);
    spans.forEach(span => {
      const spanStart = timestamp(span.start_ts), spanDuration = durationValue(span.duration_ms);
      ends.push(timestamp(span.end_ts));
      if (spanStart !== null && spanDuration !== null) ends.push(spanStart + spanDuration);
    });
    const live = runStatus(run) === "running";
    if (live && starts.length) ends.push(now);
    const end = Math.max(...ends.filter(time => time !== null));
    return { start, end, duration: Math.max(0, end - start), live };
  }
  function spanTiming(span, axis) {
    const start = timestamp(span.start_ts);
    const explicit = durationValue(span.duration_ms);
    const end = timestamp(span.end_ts);
    const duration = explicit ?? (start !== null ? Math.max(0, (end ?? axis.end) - start) : 0);
    if (start === null) return { offset: 0, width: 0, duration, known: false };
    const scale = Math.max(1, axis.duration);
    const offset = Math.min(100, Math.max(0, (start - axis.start) / scale * 100));
    return { offset, width: Math.min(100 - offset, duration / scale * 100), duration, known: true };
  }
  function orderedSpans(spans) {
    const byId = new Map(spans.map(span => [span.span_id, span]));
    const children = new Map();
    spans.forEach(span => {
      const parent = byId.has(span.parent_id) && span.parent_id !== span.span_id ? span.parent_id : null;
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(span);
    });
    const rows = [], seen = new Set();
    function visit(span, depth) {
      if (seen.has(span.span_id)) return;
      seen.add(span.span_id);
      rows.push({ span, depth });
      (children.get(span.span_id) || []).forEach(child => visit(child, depth + 1));
    }
    (children.get(null) || []).forEach(span => visit(span, 0));
    spans.forEach(span => visit(span, 0));
    return rows;
  }
  function runCounts(run) {
    const spans = run.spans || [];
    const counts = run.counts || {};
    return {
      atoms: counts.atoms ?? spans.filter(span => /^(atom:|tiptop:step)/.test(span.name || "")).length,
      llm_calls: counts.llm_calls ?? spans.filter(span => (span.name || "").startsWith("llm:")).length,
      react_iters: counts.react_iters ?? spans.filter(span => span.name === "react").length,
    };
  }
  function artifactURL(runId, path) {
    const parts = String(path || "").split("/");
    if (parts[0] !== "artifacts" || parts[1] !== String(runId) || parts.length < 3) return null;
    const suffix = parts.slice(2);
    if (suffix.some(part => !part || part === "." || part === ".." || /[\\\u0000-\u001f]/.test(part))) return null;
    return `/api/runs/${encodeURIComponent(runId)}/artifacts/${suffix.map(encodeURIComponent).join("/")}`;
  }
  function filterLogs(logs, { level = "", text = "", spanIds = null } = {}) {
    const needle = text.trim().toLowerCase();
    return logs.filter(log => (!level || String(log.level || "info").toLowerCase() === level)
      && (!spanIds || spanIds.has(log.span_id))
      && (!needle || [log.message, log.logger, log.exc_type, log.span_id, log.seq].join(" ").toLowerCase().includes(needle)));
  }
  function durationLabel(ms) {
    if (!Number.isFinite(ms)) return "—";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
    return `${Math.floor(ms / 60000)} m ${Math.floor(ms % 60000 / 1000)} s`;
  }
  class LatestRequest {
    constructor() { this.version = 0; this.controller = null; }
    cancel() { this.version++; this.controller?.abort(); }
    begin() {
      this.cancel();
      this.controller = new AbortController();
      const version = this.version;
      return { signal: this.controller.signal, current: () => version === this.version };
    }
  }
  const api = { agentName, taskName, runStatus, statusName, statusTone, completionLabel, filterRuns,
    timestamp, timeline, spanTiming, orderedSpans, runCounts, artifactURL, filterLogs, durationLabel, LatestRequest };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TraceUI = api;
})(globalThis);
