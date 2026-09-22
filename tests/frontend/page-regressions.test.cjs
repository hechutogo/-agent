const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const TraceUI = require("../../src/pickparts_agent/interfaces/static/ui.js");

// Minimal DOM boundary: run the actual page scripts, including rendering and listeners.
class Element {
  constructor(tag = "div", id = "") {
    Object.assign(this, { tagName: tag.toUpperCase(), id, children: [], dataset: {}, style: {},
      attributes: {}, listeners: {}, value: "", className: "", hidden: false, open: false,
      clientHeight: 200, clientWidth: 300, clientLeft: 0, scrollHeight: 200, _scrollTop: 0 });
    this.classList = { toggle: (name, enabled) => {
      const names = new Set(this.className.split(" ").filter(Boolean));
      if (enabled) names.add(name); else names.delete(name);
      this.className = [...names].join(" ");
    } };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return (this._text || "") + this.children.map(child => child.textContent).join(""); }
  append(...nodes) { nodes.forEach(node => this.insertBefore(node, null)); }
  insertBefore(node, before) {
    node.remove();
    this.children.splice(before ? this.children.indexOf(before) : this.children.length, 0, node);
    node.parent = this;
  }
  remove() {
    if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1);
    this.parent = null;
  }
  replaceChildren(...nodes) {
    [...this.children].forEach(node => node.remove());
    this._text = "";
    this.append(...nodes);
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  async emit(name, event = {}) {
    for (const handler of this.listeners[name] || []) {
      await handler({ target: this, preventDefault() {}, ...event });
    }
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (selector === '[aria-current="step"]' ? child.attributes["aria-current"] === "step"
        : selector.startsWith(".") ? child.className.split(" ").includes(selector.slice(1))
          : child.tagName.toLowerCase() === selector) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  getBoundingClientRect() {
    const top = this.parent?.id === "workflow-lane"
      ? this.parent.children.indexOf(this) * 80 - this.parent.scrollTop : 0;
    return { top, bottom: top + (this.parent?.id === "workflow-lane" ? 80 : this.clientHeight),
      left: 0, right: this.clientWidth + 15 };
  }
  get scrollTop() { return this._scrollTop; }
  set scrollTop(value) {
    this._scrollTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight));
    queueMicrotask(() => this.emit("scroll"));
  }
  getContext() { return {}; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
}

function page(script) {
  const elements = new Map(), requests = [];
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element("div", id));
    return elements.get(id);
  };
  const sandbox = vm.createContext({
    TraceUI, URL, AbortController, AbortSignal, console,
    document: { getElementById: get, createElement: tag => new Element(tag),
      querySelectorAll: () => [], addEventListener() {}, hidden: false },
    window: { addEventListener() {} },
    location: { href: "http://localhost/history?day=2026-09-21" },
    history: { replaceState() {}, pushState() {} },
    setTimeout: () => 0, clearTimeout() {},
    fetch: (url, options) => new Promise((resolve, reject) => requests.push({ url, options, resolve, reject })),
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,
    `../../src/pickparts_agent/interfaces/static/${script}.js`), "utf8"), sandbox);
  return { get, requests, run: code => vm.runInContext(code, sandbox) };
}

function respond(request, data, status = 200) {
  request.resolve({ ok: status >= 200 && status < 300, status, json: async () => data });
}
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const ready = extra => ({ status: "ready", agent: "pickparts", frame_id: 0,
  messages: [], run_id: "previous", run_task: "command", ...extra });
async function consolePage(initial = ready()) {
  const app = page("app");
  respond(app.requests.shift(), initial);
  await flush();
  return app;
}

test("delayed running list cannot undo terminal detail or its filter membership", async () => {
  const history = page("history");
  respond(history.requests.shift(), { runs: [{ run_id: "r1", status: "running" }] });
  await flush();
  respond(history.requests.shift(), { run_id: "r1", status: "running" });
  await flush();
  history.get("run-status").value = "success";
  const list = history.run("loadRuns()");
  const detail = history.run("refreshDetail()");
  respond(history.requests[1], { run_id: "r1", status: "success", message: "finished",
    end_ts: "2026-09-21T01:00:00Z" });
  await detail;
  assert.equal(history.get("run-list").children.length, 1);
  respond(history.requests[0], { runs: [{ run_id: "r1", status: "running" }] });
  await list;
  assert.equal(history.get("run-list").children.length, 1);
  assert.match(history.get("run-list").textContent, /已完成/);
  assert.equal(history.get("list-notice").hidden, true);
  history.get("run-status").value = "running";
  await history.get("run-status").emit("change");
  assert.equal(history.get("run-list").children.length, 0);
});

function workflow(current) {
  return { goal: "long plan", status: "running", current,
    steps: Array.from({ length: 30 }, (_, index) => ({
      index, atom: "move", args: { target: index },
      status: index < current ? "done" : index === current ? "active" : "pending",
    })) };
}

test("long-plan programmatic scrolling keeps following; user scrolling pauses until current-step", async () => {
  const app = await consolePage(), lane = app.get("workflow-lane");
  lane.scrollHeight = 2400;
  app.run(`renderWorkflow(${JSON.stringify(workflow(8))})`);
  await flush();
  assert.equal(lane.scrollTop, 520);
  app.run(`renderWorkflow(${JSON.stringify(workflow(9))})`);
  await flush();
  assert.equal(lane.scrollTop, 600, "programmatic scroll must not suspend follow");
  await lane.emit("wheel", { deltaY: -200 });
  lane.scrollTop = 100;
  await flush();
  app.run(`renderWorkflow(${JSON.stringify(workflow(10))})`);
  assert.equal(lane.scrollTop, 100, "explicit user scrolling suspends follow");
  await app.get("workflow-current").emit("click");
  await flush();
  assert.equal(lane.scrollTop, 680);
  app.run(`renderWorkflow(${JSON.stringify(workflow(11))})`);
  await flush();
  assert.equal(lane.scrollTop, 760, "current-step resumes persistent follow");
  assert.equal(lane.querySelector(".wf-label").textContent, "移动");
  const parameters = lane.querySelector(".wf-parameters");
  parameters.open = true;
  app.run(`renderWorkflow(${JSON.stringify(workflow(11))})`);
  assert.equal(lane.querySelector(".wf-parameters"), parameters);
  assert.equal(parameters.open, true);
});

for (const [name, event] of [
  ["touchmove", {}], ["keydown", { key: "PageUp" }], ["pointerdown", { clientX: 310, clientY: 50 }],
]) {
  test(`workflow ${name} navigation suspends following`, async () => {
    const app = await consolePage(), lane = app.get("workflow-lane");
    lane.scrollHeight = 2400;
    await lane.emit(name, event);
    app.run(`renderWorkflow(${JSON.stringify(workflow(8))})`);
    assert.equal(lane.scrollTop, 0);
  });
}

for (const failure of ["network", "timeout"]) {
  test(`${failure} POST outcome stays uncertain and retains input after fresh state`, async () => {
    const app = await consolePage();
    app.get("instruction").value = "move block";
    const submission = app.get("command-form").emit("submit");
    app.requests.shift().reject(Object.assign(new Error("connection lost"),
      { name: failure === "timeout" ? "TimeoutError" : "TypeError" }));
    await flush();
    assert.equal(app.get("dialog").open, false, "wait for reconciliation before showing an outcome");
    assert.equal(app.get("send").disabled, true);
    assert.equal(app.requests[0].url, "/api/state");
    respond(app.requests.shift(), ready());
    await submission;
    assert.equal(app.get("instruction").value, "move block");
    assert.match(app.get("dialog-title").textContent, /待确认/);
    assert.match(app.get("dialog-text").textContent, /可能已受理.*勿重复提交/);
    assert.doesNotMatch(app.get("dialog-title").textContent, /未发送|失败/);
  });
}

test("lost command response with a newly appended matching user message clears input", async () => {
  const previous = { role: "user", text: "move block", agent: "pickparts" };
  const app = await consolePage(ready({ messages: [previous] }));
  app.get("instruction").value = "move block";
  const submission = app.get("command-form").emit("submit");
  app.requests.shift().reject(new Error("connection lost"));
  await flush();
  respond(app.requests.shift(), ready({ status: "busy", messages: [previous, { ...previous }] }));
  await submission;
  assert.equal(app.get("instruction").value, "");
  assert.doesNotMatch(app.get("dialog-title").textContent, /未发送|失败|待确认/);
});

test("an old matching message plus unrelated busy run does not confirm acceptance", async () => {
  const messages = [{ role: "user", text: "move block", agent: "pickparts" }];
  const app = await consolePage(ready({ messages }));
  app.get("instruction").value = "move block";
  const submission = app.get("command-form").emit("submit");
  app.requests.shift().reject(new Error("connection lost"));
  await flush();
  respond(app.requests.shift(), ready({ status: "busy", messages, run_id: "unrelated",
    workflow: { goal: "another command", steps: [] } }));
  await submission;
  assert.equal(app.get("instruction").value, "move block");
  assert.match(app.get("dialog-title").textContent, /待确认/);
});

test("failed state reconciliation leaves transport outcome uncertain and controls offline", async () => {
  const app = await consolePage();
  app.get("instruction").value = "move block";
  const submission = app.get("command-form").emit("submit");
  app.requests.shift().reject(new Error("connection lost"));
  await flush();
  app.requests.shift().reject(new Error("still offline"));
  await submission;
  assert.equal(app.get("instruction").value, "move block");
  assert.equal(app.get("send").disabled, true);
  assert.match(app.get("dialog-title").textContent, /待确认/);
});

for (const malformed of [false, true]) {
  test(`HTTP rejection remains explicit with ${malformed ? "non-JSON" : "JSON"} error body`, async () => {
    const app = await consolePage();
    app.get("instruction").value = "move block";
    const submission = app.get("command-form").emit("submit");
    const request = app.requests.shift();
    if (malformed) request.resolve({ ok: false, status: 409, json: async () => { throw new SyntaxError("not JSON"); } });
    else respond(request, { detail: "busy, rejected" }, 409);
    await flush();
    respond(app.requests.shift(), ready());
    await submission;
    assert.equal(app.get("instruction").value, "move block");
    assert.match(app.get("dialog-title").textContent, /未发送|未受理/);
    assert.match(app.get("dialog-text").textContent, malformed ? /409/ : /busy, rejected/);
    assert.doesNotMatch(app.get("dialog-text").textContent, /可能已受理/);
  });
}

test("optimized subtasks update independently of current motion steps and clear on switch", async () => {
  const steps = [
    { index: 0, instruction: "stack A", status: "active" },
    { index: 1, instruction: "<b>store B</b>", status: "pending" },
  ];
  const snapshot = ready({ agent: "tiptop_optimized", status: "busy",
    subtasks: { steps, current: 0 }, workflow: workflow(0),
    models: { llm: "test", vision: "本地 RGB-D" },
    react: [{ attempt: 1, decision: "retry", detail: "retry grasp" }] });
  const app = await consolePage(snapshot), lane = app.get("subtask-lane");
  assert.equal(app.get("subtask-panel").hidden, false);
  assert.equal(lane.children.length, 2);
  assert.match(app.get("subtask-progress").textContent, /0 \/ 2.*当前 1/);
  assert.match(lane.querySelector('[aria-current="step"]').textContent, /stack A.*执行中/);
  assert.equal(lane.children[1].querySelector("b"), null, "instructions are text, not HTML");
  assert.match(lane.children[1].textContent, /<b>store B<\/b>/);
  assert.match(app.get("workflow-progress").textContent, /0 \/ 30/);
  assert.match(app.get("react-title").textContent, /ReAct/);
  assert.match(app.get("react-lane").textContent, /retry grasp/);
  assert.match(app.get("agent-mode").textContent, /子任务/);
  assert.doesNotMatch(app.get("agent-mode").textContent, /开环/);
  assert.doesNotMatch(app.get("model-label").textContent, /云端视觉/);
  assert.equal(app.get("agent-select").children.find(option => option.value === "tiptop_optimized").textContent, "TiPToP 优化版");

  steps[0].status = "done";
  steps[1].status = "active";
  snapshot.subtasks.current = 1;
  snapshot.workflow = { goal: "store B", current: 0, status: "running",
    steps: [{ index: 0, atom: "gripper", args: {}, status: "active" }] };
  app.run(`render(${JSON.stringify(snapshot)})`);
  assert.equal(lane.children.length, 2, "full snapshots replace rows instead of appending");
  assert.match(app.get("subtask-progress").textContent, /1 \/ 2.*当前 2/);
  assert.match(lane.querySelector('[aria-current="step"]').textContent, /store B/);
  assert.match(app.get("workflow-progress").textContent, /0 \/ 1.*夹爪/);
  assert.doesNotMatch(app.get("workflow-lane").textContent, /stack A|store B/);

  steps[1].status = "failed";
  snapshot.status = "ready";
  snapshot.messages = [{ role: "assistant", agent: "tiptop_optimized", success: false,
    verification: "visual", text: "stopped", completed_subtasks: 1, total_subtasks: 2 }];
  app.run(`render(${JSON.stringify(snapshot)})`);
  assert.match(lane.children[1].textContent, /失败/);
  assert.equal(lane.querySelector('[aria-current="step"]'), null);
  assert.match(app.get("messages").textContent, /未完成/);
  app.run(`render(${JSON.stringify(ready({ agent: "tiptop", subtasks: null }))})`);
  assert.equal(app.get("subtask-panel").hidden, true);
  assert.equal(lane.children.length, 0);
  assert.match(app.get("agent-mode").textContent, /开环/);
  assert.match(app.get("react-title").textContent, /开环/);
});

test("optimized history filter and verification remain distinct from original TiPToP", async () => {
  const history = page("history");
  respond(history.requests.shift(), { runs: [
    { run_id: "optimized", agent: "tiptop_optimized", status: "success", task: "command" },
    { run_id: "original", agent: "tiptop", status: "success", task: "command" },
  ] });
  await flush();
  respond(history.requests.shift(), { run_id: "optimized", agent: "tiptop_optimized",
    status: "success", task: "command", message: "verified" });
  await flush();
  assert.equal(history.get("run-agent-label").textContent, "TiPToP 优化版");
  assert.equal(history.get("run-verification").hidden, false);
  assert.match(history.get("run-verification").textContent, /视觉/);
  assert.doesNotMatch(history.get("run-verification").textContent, /开环/);
  history.get("run-agent").value = "tiptop_optimized";
  await history.get("run-agent").emit("change");
  assert.equal(history.get("run-list").children.length, 1);
  assert.match(history.get("run-list").textContent, /TiPToP 优化版/);
  const original = history.run('selectRun("original")');
  respond(history.requests.shift(), { run_id: "original", agent: "tiptop",
    status: "success", task: "command" });
  await original;
  assert.equal(history.get("run-agent-label").textContent, "TiPToP 初版");
  assert.match(history.get("run-verification").textContent, /开环.*不代表目标已通过视觉验证/);
});

test("subtask current-only snapshots refresh the highlight and completed tasks have no active row", async () => {
  const steps = [
    { index: 0, instruction: "stack A", status: "pending" },
    { index: 1, instruction: "store B", status: "pending" },
  ];
  const app = await consolePage(ready({ agent: "tiptop_optimized",
    subtasks: { steps, current: 0 } }));
  const lane = app.get("subtask-lane");
  app.run(`renderSubtasks(${JSON.stringify({ steps, current: 1 })})`);
  assert.equal(lane.children[0].attributes["aria-current"], undefined);
  assert.equal(lane.children[1].attributes["aria-current"], "step");
  assert.match(lane.children[1].textContent, /执行中/);
  const current = lane.children[1];
  app.run(`renderSubtasks(${JSON.stringify({ steps, current: 1 })})`);
  assert.equal(lane.children[1], current, "unchanged polling keeps the same DOM row");
  steps.forEach(step => { step.status = "done"; });
  app.run(`render(${JSON.stringify(ready({ agent: "tiptop_optimized",
    subtasks: { steps, current: 2 },
    messages: [{ role: "assistant", agent: "tiptop_optimized", text: "done",
      success: true, verification: "visual" }] }))})`);
  assert.equal(lane.querySelector('[aria-current="step"]'), null);
  assert.match(app.get("subtask-progress").textContent, /2 \/ 2/);
  assert.match(app.get("messages").textContent, /视觉确认完成/);
  app.run("renderSubtasks({steps: [], current: null})");
  assert.equal(lane.children.length, 0);
  assert.equal(app.get("subtask-panel").hidden, true);
});
