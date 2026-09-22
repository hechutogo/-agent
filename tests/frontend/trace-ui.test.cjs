const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const file = path.resolve(__dirname, "../../src/pickparts_agent/interfaces/static/ui.js");
const ui = fs.existsSync(file) ? require(file) : {};

test("TiPToP success never implies visual verification", () => {
  assert.equal(typeof ui.completionLabel, "function");
  assert.match(ui.completionLabel({ success: true, agent: "tiptop", verification: "open_loop" }), /开环/);
  assert.doesNotMatch(ui.completionLabel({ success: true, agent: "tiptop", verification: "visual" }), /视觉确认/);
  assert.match(ui.completionLabel({ success: true, agent: "pickparts", verification: "visual" }), /视觉确认/);
  assert.doesNotMatch(ui.completionLabel({ success: true }), /视觉确认/);
  assert.match(ui.completionLabel({ success: false, verification: "visual" }), /未完成/);
});

test("optimized visual completion and history name filtering do not inherit open-loop semantics", () => {
  assert.equal(ui.completionLabel({ success: true, agent: "tiptop_optimized", verification: "visual" }), "视觉确认完成");
  assert.match(ui.completionLabel({ success: false, agent: "tiptop_optimized", verification: "visual" }), /未完成/);
  const runs = [
    { run_id: "v1", agent: "tiptop", status: "success" },
    { run_id: "v2", agent: "tiptop_optimized", status: "success" },
  ];
  assert.deepEqual(ui.filterRuns(runs, { text: "优化版" }).map(run => run.run_id), ["v2"]);
  assert.deepEqual(ui.filterRuns(runs, { text: "初版" }).map(run => run.run_id), ["v1"]);
  assert.deepEqual(ui.filterRuns(runs, { agent: "tiptop_optimized", status: "success" }).map(run => run.run_id), ["v2"]);
});

test("status filtering normalizes missing live status and terminal aliases", () => {
  assert.equal(typeof ui.runStatus, "function");
  assert.equal(ui.runStatus({}), "running");
  assert.equal(ui.runStatus({ status: "ok" }), "success");
  assert.equal(ui.runStatus({ status: "failed" }), "error");
  assert.equal(ui.runStatus({ status: "aborted" }), "aborted");
});

test("incomplete results remain distinct from exceptions and switch tasks have a readable label", () => {
  assert.equal(ui.statusName("incomplete"), "未完成");
  assert.equal(ui.statusName("error"), "异常");
  assert.equal(ui.statusTone("incomplete"), "error");
  assert.equal(ui.taskName("switch"), "切换 Agent");
  assert.deepEqual(ui.filterRuns([
    { run_id: "a", status: "incomplete" }, { run_id: "b", status: "error" },
  ], { status: "incomplete" }).map(run => run.run_id), ["a"]);
});

test("agent, status and case-insensitive text filters compose without changing the source", () => {
  assert.equal(typeof ui.filterRuns, "function");
  const runs = [
    { run_id: "r1", agent: "tiptop", status: "success", goal: "Move BLOCK_A" },
    { run_id: "r2", agent: "pickparts", status: "success", goal: "Move BLOCK_A" },
    { run_id: "r3", agent: "tiptop", status: "error", goal: "Move BLOCK_A" },
  ];
  assert.deepEqual(ui.filterRuns(runs, { agent: "tiptop", status: "success", text: "block_a" }).map(r => r.run_id), ["r1"]);
  assert.equal(runs.length, 3);
  assert.equal(ui.filterRuns(runs, { text: "nonexistent" }).length, 0);
});

test("waterfall aligns sibling bars on the same absolute timeline", () => {
  assert.equal(typeof ui.timeline, "function");
  const run = {
    start_ts: "2026-09-21T00:00:00Z", duration_ms: 10000, status: "success",
    spans: [
      { span_id: "a", start_ts: "2026-09-21T00:00:02Z", duration_ms: 3000 },
      { span_id: "b", start_ts: "2026-09-21T00:00:06Z", end_ts: "2026-09-21T00:00:08Z" },
    ],
  };
  const axis = ui.timeline(run);
  assert.equal(axis.duration, 10000);
  assert.deepEqual(ui.spanTiming(run.spans[0], axis), { offset: 20, width: 30, duration: 3000, known: true });
  assert.deepEqual(ui.spanTiming(run.spans[1], axis), { offset: 60, width: 20, duration: 2000, known: true });
});

test("unfinished spans extend to now while finished spans keep their duration", () => {
  assert.equal(typeof ui.timeline, "function");
  const run = { start_ts: "2026-09-21T00:00:00Z", status: "running", spans: [] };
  const axis = ui.timeline(run, Date.parse("2026-09-21T00:00:10Z"));
  assert.equal(axis.duration, 10000);
  assert.equal(ui.spanTiming({ start_ts: "2026-09-21T00:00:02Z" }, axis).duration, 8000);
  assert.equal(ui.spanTiming({ start_ts: "2026-09-21T00:00:02Z", duration_ms: 20 }, axis).duration, 20);
  assert.equal(ui.spanTiming({}, axis).known, false);
});

test("tree ordering nests children even when events arrive out of order and terminates cycles", () => {
  assert.equal(typeof ui.orderedSpans, "function");
  const result = ui.orderedSpans([
    { span_id: "child", parent_id: "root" },
    { span_id: "orphan", parent_id: "missing" },
    { span_id: "root" },
    { span_id: "x", parent_id: "y" },
    { span_id: "y", parent_id: "x" },
  ]);
  assert.equal(result.length, 5);
  assert.equal(new Set(result.map(row => row.span.span_id)).size, 5);
  const root = result.findIndex(row => row.span.span_id === "root");
  assert.equal(result[root + 1].span.span_id, "child");
  assert.equal(result[root + 1].depth, 1);
});

test("metrics derive live counts and respect authoritative zero counts", () => {
  assert.equal(typeof ui.runCounts, "function");
  const spans = [{ name: "atom:grasp" }, { name: "atom:lift" }, { name: "llm:plan" }, { name: "react" }];
  assert.deepEqual(ui.runCounts({ spans }), { atoms: 2, llm_calls: 1, react_iters: 1 });
  assert.equal(ui.runCounts({ spans, counts: { atoms: 0 } }).atoms, 0);
});

test("artifact links are scoped to the run and encode names without allowing traversal", () => {
  assert.equal(typeof ui.artifactURL, "function");
  assert.equal(ui.artifactURL("r1", "artifacts/r1/llm/input #1.txt"), "/api/runs/r1/artifacts/llm/input%20%231.txt");
  assert.equal(ui.artifactURL("r1", "artifacts/r2/private.txt"), null);
  assert.equal(ui.artifactURL("r1", "artifacts/r1/../private.txt"), null);
  assert.equal(ui.artifactURL("r1", "javascript:alert(1)"), null);
});

test("new selections invalidate old responses even when abort is ignored by transport", () => {
  assert.equal(typeof ui.LatestRequest, "function");
  const gate = new ui.LatestRequest();
  const first = gate.begin();
  const second = gate.begin();
  assert.equal(first.signal.aborted, true);
  assert.equal(first.current(), false);
  assert.equal(second.current(), true);
  gate.cancel();
  assert.equal(second.current(), false);
});

test("log filters combine exact level, search, and selected span subtree", () => {
  assert.equal(typeof ui.filterLogs, "function");
  const logs = [
    { seq: 1, level: "INFO", message: "Grasp ready", span_id: "a" },
    { seq: 2, level: "error", message: "Grasp failed", span_id: "b" },
    { seq: 3, level: "error", message: "Grasp failed", span_id: "c" },
  ];
  assert.deepEqual(ui.filterLogs(logs, { level: "error", text: "GRASP", spanIds: new Set(["a", "b"]) }).map(l => l.seq), [2]);
});
