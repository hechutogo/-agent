const $ = id => document.getElementById(id);
const ATOM_LABEL = {
  find_object: "定位", reach_above: "移动到上方", grasp: "抓取",
  lift: "抬起", carry_to: "搬运", place_on: "叠放", release_into: "放入",
  place_on_table: "放到桌面", verify_state: "校验",
  reset_arm: "复位", set_gripper: "夹爪",
  move: "移动", gripper: "夹爪", calibrate: "抓取标定", reset: "复位",
};
const ACTIVITY_LABEL = {
  understanding: "理解指令", plan: "动作计划", action: "执行动作",
  observation: "观测结果", recovery: "纠偏摘要",
};
const DECISION_LABEL = {
  retry: "重新尝试", replace: "替换动作", replan: "重新规划",
  ask_user: "请求确认", abort: "安全中止",
};
const STEP_LABEL = { pending: "待执行", active: "执行中", done: "已完成", failed: "失败", skipped: "已跳过" };
let state = null, lastFrame = -1, started = null, frameBusy = false;
let reportedError = null, pending = null, connected = false, stateEpoch = 0;
let readSequence = 0, appliedSequence = 0;
let activeStepKey = null;
let followWorkflow = true;
const context = $("camera").getContext("2d");
const lists = new WeakMap();

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Preserve unchanged rows across polling, including histories trimmed by the server.
function reconcile(lane, entries, keyFor, create) {
  const previous = lists.get(lane) || new Map();
  const next = new Map();
  let changed = false;
  entries.forEach((entry, index) => {
    const key = keyFor(entry, index);
    const signature = JSON.stringify(entry);
    const cached = previous.get(key);
    const row = cached?.signature === signature ? cached.node : create(entry, index);
    if (lane.children[index] !== row) {
      lane.insertBefore(row, lane.children[index] || null);
      changed = true;
    }
    next.set(key, { node: row, signature });
  });
  for (const [key, cached] of previous) {
    if (next.get(key)?.node !== cached.node) {
      cached.node.remove();
      changed = true;
    }
  }
  lists.set(lane, next);
  return changed;
}

function followLatest(lane, button) {
  let following = true;
  function updateButton() {
    button.hidden = following || lane.scrollHeight <= lane.clientHeight + 24;
  }
  function latest() {
    following = true;
    lane.scrollTop = lane.scrollHeight;
    updateButton();
  }
  lane.addEventListener("scroll", () => {
    following = lane.scrollHeight - lane.clientHeight - lane.scrollTop <= 24;
    updateButton();
  }, { passive: true });
  button.addEventListener("click", () => { latest(); lane.focus({ preventScroll: true }); });
  return changed => {
    if (changed && following) latest();
    else updateButton();
  };
}

const followReact = followLatest($("react-lane"), $("react-latest"));
const followChat = followLatest($("chat-scroll"), $("chat-latest"));

function canMutate(kind) {
  if (!connected || pending || !state) return false;
  return kind === "reset"
    ? ["ready", "error"].includes(state.status)
    : state.status === "ready" && !state.recovery_required;
}

function renderControls() {
  const ready = canMutate("command");
  $("agent-select").disabled = !canMutate("agent");
  $("instruction").disabled = $("send").disabled = !ready;
  $("add-block").disabled = $("add-box").disabled = !ready;
  $("reset").disabled = !canMutate("reset");
  document.querySelectorAll("[data-command]").forEach(button => { button.disabled = !ready; });
  $("add-block").textContent = pending === "block" ? "添加中…" : "+ 添加方块";
  $("add-box").textContent = pending === "box" ? "添加中…" : "+ 添加盒子";
  $("reset").textContent = pending === "reset" || state?.status === "resetting" ? "重置中…" : "↻ 随机重置";
  $("scene-panel").setAttribute("aria-busy", String(Boolean(pending) || ["starting", "busy", "adding", "resetting", "switching"].includes(state?.status)));
  $("agent-switch-note").textContent = pending === "agent" || state?.status === "switching"
    ? "正在切换执行引擎，请稍候。当前场景保持不变。"
    : state?.recovery_required ? "需先重置场景恢复安全状态，才能切换 Agent。"
      : "切换保留当前场景，清空对话与动作进程；历史记录保留。";
}

function renderAgent() {
  const agent = state.agent || "pickparts";
  const tiptop = agent === "tiptop";
  const optimized = agent === "tiptop_optimized";
  const agents = state.agents?.length ? state.agents : [
    { id: "pickparts", name: "自研 PickParts" }, { id: "tiptop", name: "TiPToP 初版" },
    { id: "tiptop_optimized", name: "TiPToP 优化版" },
  ];
  reconcile($("agent-select"), agents, item => item.id, item => {
    const option = element("option", "", item.name || TraceUI.agentName(item.id));
    option.value = item.id;
    return option;
  });
  $("agent-select").value = agent;
  $("agent-mode").textContent = tiptop ? "开环 · 一次规划" : optimized ? "子任务闭环 · ReAct" : "闭环 · ReAct";
  $("agent-mode").classList.toggle("open-loop", tiptop);
  $("agent-description").textContent = agents.find(item => item.id === agent)?.description
    || (tiptop ? "本地 RGB-D 感知、TAMP 一次性规划与开环执行；不进行 ReAct 纠偏或视觉结果验证。"
      : optimized ? "将指令拆分为有序子任务，逐个感知、规划与执行，通过视觉校验与 ReAct 纠偏确认结果。"
        : "语言规划、视觉感知与 ReAct 纠偏，以视觉观测校验结果。");
  $("react-title").textContent = tiptop ? "开环执行说明" : "ReAct 纠偏记录";
  $("react-description").textContent = tiptop
    ? "TiPToP 按一次性计划执行，不启用 ReAct；动作完成不代表目标已通过视觉验证。"
    : "记录异常、重试与方案调整，自动跟随最新进展。";
  $("react-empty").textContent = tiptop ? "此模式不产生 ReAct 纠偏记录" : "暂无纠偏记录";
  $("welcome-description").textContent = tiptop
    ? "描述要叠放或收纳的物体。TiPToP 将结合本地 RGB-D 感知生成一次性计划，并反馈开环执行进展。"
    : optimized ? "可一次描述多个有序目标。我会逐个完成子任务，展示当前动作、视觉校验与纠偏进展。"
      : "告诉我把物体叠放到哪里，或放进哪个盒子。我会反馈理解、计划、动作与观测摘要。";
  $("pipeline-label").textContent = tiptop
    ? "本地 RGB-D 感知 → TAMP 规划 → 开环执行（未视觉验证）"
    : optimized ? "子任务拆解 → 本地 RGB-D 感知 → 规划执行 → 视觉校验 / ReAct 纠偏"
      : "语言规划 → 视觉感知 → 物理执行 → 结果验证";
  $("motion-title").textContent = optimized ? "当前子任务 · 动作进程" : "动作进程";
  $("current-run").hidden = !state.run_id;
  $("current-run").href = state.run_id ? `/history?run=${encodeURIComponent(state.run_id)}` : "/history";
  $("current-run-title").textContent = `${TraceUI.agentName(agent)} · ${TraceUI.taskName(state.run_task)}`;
  $("current-run-id").textContent = state.run_id || "";
}

function showDialog(title, text, confirm = null) {
  $("dialog-title").textContent = title;
  $("dialog-text").textContent = text;
  $("dialog-cancel").hidden = !confirm;
  $("dialog-ok").textContent = confirm ? "重置场景" : "知道了";
  $("dialog-ok").onclick = () => { $("dialog").close(); if (confirm) confirm(); };
  $("dialog-cancel").onclick = () => $("dialog").close();
  if (!$("dialog").open) $("dialog").showModal();
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    const result = await response.json().catch(() => null);
    const error = Error(typeof result?.detail === "string" ? result.detail : `请求未受理（HTTP ${response.status}）。`);
    error.rejected = true;
    throw error;
  }
  return response.json();
}

function renderMessages(messages = []) {
  $("welcome").hidden = messages.length > 0;
  const changed = reconcile($("messages"), messages,
    (message, index) => `${message.role}:${message.seq ?? index}`, message => {
    const activity = message.role === "activity";
    const kind = Object.hasOwn(ACTIVITY_LABEL, message.kind) ? message.kind : "action";
    const roleName = activity ? "activity" : message.role === "user" ? "user" : "assistant";
    const article = element("article", `message ${roleName}${activity ? ` ${kind}` : ""}`);
    if (message.seq != null) article.dataset.seq = String(message.seq);
    const role = element("div", "role", activity
      ? `${ACTIVITY_LABEL[kind]} · ${TraceUI.agentName(message.agent || state?.agent || "pickparts")}`
      : roleName === "user" ? "你" : TraceUI.agentName(message.agent || state?.agent || "pickparts"));
    if (activity && message.seq != null) role.append(element("span", "activity-seq", `#${message.seq}`));
    const content = element("div", "content", message.text || "");
    article.append(role, content);
    if (message.role === "assistant" && typeof message.success === "boolean") {
      article.append(element("span", "result-tag" + (message.success ? "" : " failed"),
        TraceUI.completionLabel({ ...message, agent: message.agent || state?.agent || "pickparts" })));
    }
    return article;
  });
  followChat(changed);
}

function renderSubtasks(subtasks) {
  const steps = subtasks?.steps || [];
  const active = steps.find(step => step.status === "active")
    || steps.find(step => step.index === subtasks?.current && step.status === "pending");
  const done = steps.filter(step => step.status === "done").length;
  $("subtask-panel").hidden = !steps.length;
  $("subtask-progress").textContent = steps.length
    ? `已完成 ${done} / ${steps.length}${active ? ` · 当前 ${active.index + 1}：${active.instruction}` : ""}`
    : "尚无子任务";
  reconcile($("subtask-lane"), steps.map(step => ({ ...step, status: step === active ? "active" : step.status })),
    step => step.index, step => {
    const status = Object.hasOwn(STEP_LABEL, step.status) ? step.status : "pending";
    const row = element("li", `subtask-row ${status}`);
    row.value = step.index + 1;
    if (status === "active") row.setAttribute("aria-current", "step");
    row.append(element("span", "subtask-instruction", step.instruction),
      element("span", "subtask-status", STEP_LABEL[status]));
    return row;
  });
}

function renderWorkflow(wf) {
  const lane = $("workflow-lane");
  const steps = (wf?.steps || []).map((step, index) => ({ ...step, index: step.index ?? index }));
  const active = steps.find(step => step.status === "active" || step.status === "running")
    || (wf?.status === "running" ? steps.find(step => step.index === wf.current && step.status === "pending") : null);
  const done = steps.filter(step => step.status === "done").length;
  $("workflow-goal").textContent = wf?.goal || "等待可执行的工作流…";
  $("workflow-progress").textContent = steps.length
    ? `已完成 ${done} / ${steps.length}${active ? ` · 当前 ${active.index + 1}：${ATOM_LABEL[active.atom] || active.atom}` : ""}`
    : "尚无动作";
  $("workflow-current").disabled = !active;
  reconcile(lane, steps.map(step => ({ ...step, status: step === active ? "active" : step.status })),
    step => step.index, step => {
    const status = Object.hasOwn(STEP_LABEL, step.status) ? step.status : "pending";
    const node = element("div", `wf-node ${status}`);
    node.setAttribute("role", "listitem");
    if (status === "active") node.setAttribute("aria-current", "step");
    node.append(element("b", "wf-index", String(step.index + 1).padStart(2, "0")));
    const detail = element("div", "wf-detail");
    const heading = element("div", "wf-heading");
    heading.append(element("span", "wf-label", ATOM_LABEL[step.atom] || step.atom),
      element("span", "wf-status", STEP_LABEL[status]));
    const call = element("details", "wf-parameters");
    call.append(element("summary", "", "动作参数"),
      element("code", "wf-call", `${step.atom}(${JSON.stringify(step.args || {})})`));
    detail.append(heading, call);
    node.append(detail);
    return node;
  });
  const key = active ? `${wf.goal}:${active.index}:${active.atom}` : null;
  if (key !== activeStepKey && active && followWorkflow) revealActiveStep();
  activeStepKey = key;
}

function revealActiveStep() {
  const lane = $("workflow-lane");
  const active = lane.querySelector('[aria-current="step"]');
  if (!active) return;
  const row = active.getBoundingClientRect(), bounds = lane.getBoundingClientRect();
  if (row.top < bounds.top) lane.scrollTop += row.top - bounds.top;
  else if (row.bottom > bounds.bottom) lane.scrollTop += row.bottom - bounds.bottom;
}

function renderLegend(objects = []) {
  $("legend-empty").hidden = objects.length > 0;
  reconcile($("scene-objects"), objects, object => object.id, object => {
    const item = element("span", "scene-object");
    item.setAttribute("role", "listitem");
    const color = Array.isArray(object.color) ? object.color : [];
    const clamp = (value, fallback) => Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
    const rgb = [0, 1, 2].map(index => Math.round(clamp(color[index], 0.5) * 255));
    const swatch = element("i", "object-swatch");
    swatch.style.backgroundColor = `rgba(${rgb.join(", ")}, ${clamp(color[3], 1)})`;
    swatch.setAttribute("aria-hidden", "true");
    const label = object.label || (object.kind === "box" ? "盒子" : "方块");
    item.append(swatch, element("span", "object-label", label), element("code", "object-id", String(object.id)));
    return item;
  });
}

function renderReact(entries = []) {
  $("react-empty").hidden = entries.length > 0;
  const summaries = entries.map(entry => ({
    seq: entry.seq, attempt: entry.attempt, decision: entry.decision,
    text: entry.summary || entry.detail || "依据执行结果调整后续动作。",
  }));
  const changed = reconcile($("react-lane"), summaries, (entry, index) => entry.seq ?? index, entry => {
    const item = element("div", "react-item");
    const decision = DECISION_LABEL[entry.decision] || entry.decision || "执行反馈";
    const tag = element("span", "react-tag", `${entry.attempt != null ? `第 ${entry.attempt} 次 · ` : ""}${decision}`);
    const text = element("p", "", String(entry.text).slice(0, 320));
    item.append(tag, text);
    return item;
  });
  followReact(changed);
}

async function mutate(kind, path, body) {
  if (!canMutate(kind)) return false;
  const before = state, input = $("instruction").value;
  const matchingMessages = snapshot => (snapshot.messages || []).filter(message =>
    message.role === "user" && message.text === body?.text
    && (message.agent || snapshot.agent) === before.agent).length;
  pending = kind;
  stateEpoch++;
  renderControls();
  let accepted = false, failure = null;
  try {
    await post(path, body);
    accepted = true;
    if (kind === "reset") lastFrame = -1;
  } catch (error) {
    failure = error;
  } finally {
    // Discard reads started before the POST completed; keep controls locked until a fresh state arrives.
    stateEpoch++;
    const fresh = await refreshState();
    if (failure && !failure.rejected && kind === "command" && fresh?.agent === before.agent) {
      // User messages have no unique ID; only an increase proves a new matching submission.
      accepted = matchingMessages(fresh) > matchingMessages(before);
    }
    if (failure && !accepted) {
      if (kind === "command") $("instruction").value = input;
      const title = kind === "agent" ? "切换失败" : kind === "reset" ? "重置失败" : kind === "command" ? "指令未发送" : "添加失败";
      showDialog(failure.rejected ? title : "提交结果待确认", failure.rejected ? failure.message
        : "连接中断，操作可能已受理，请先查看当前任务与历史记录，勿重复提交");
    }
    pending = null;
    renderControls();
  }
  return accepted;
}

async function updateFrame(id) {
  if (frameBusy || id === lastFrame || id === 0) return;
  frameBusy = true;
  try {
    const response = await fetch(`/api/frame?v=${id}`);
    if (response.status !== 200) return;
    const bitmap = await createImageBitmap(await response.blob());
    context.drawImage(bitmap, 0, 0, 640, 480);
    bitmap.close();
    lastFrame = id;
    $("camera-loading").hidden = true;
    $("frame-label").textContent = `FRAME ${String(id).padStart(5, "0")}`;
  } catch (_) {
    $("frame-label").textContent = "画面重连中";
  } finally { frameBusy = false; }
}

function render(next) {
  const wasBusy = state?.status === "busy";
  const previousAgent = state?.agent;
  state = next;
  if (previousAgent && previousAgent !== state.agent) {
    $("instruction").value = "";
    activeStepKey = null;
    followWorkflow = true;
  }
  const busy = state.status === "busy";
  const loading = ["starting", "resetting"].includes(state.status);
  if (busy && !wasBusy) started = Date.now();
  if (!busy && wasBusy && started) $("elapsed").textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`;
  if (busy && started) $("elapsed").textContent = `${Math.floor((Date.now() - started) / 1000)} s`;
  $("connection").textContent = "本地服务已连接";
  $("connection-dot").style.background = "#6aaf70";
  $("scene-status").textContent = state.status === "resetting" ? "随机重置中" : loading ? "初始化中"
    : state.status === "switching" ? "切换 Agent 中"
      : state.status === "adding" ? "添加物体中" : busy ? "执行中" : state.status === "error" ? "需要处理" : "已连接";
  $("stage-label").textContent = state.stage || "等待指令";
  renderAgent();
  renderControls();
  $("thinking").hidden = !busy && !["adding", "switching"].includes(state.status);
  $("thinking-text").textContent = `${state.stage || "处理请求"}…`;
  $("notice").hidden = !(state.error || state.recovery_required);
  $("notice").textContent = state.error || (state.recovery_required ? "操作中断，需重置场景后才能继续。" : "");
  const models = state.models || {};
  $("model-label").textContent = models.llm
    ? `${models.llm}${models.vision ? ` · ${["tiptop", "tiptop_optimized"].includes(state.agent) ? models.vision : `云端视觉 ${models.vision.replace(/^ark\//, "")}`}` : ""}`
    : "云端模型等待就绪";
  $("reasoning-label").hidden = typeof models.thinking !== "boolean";
  $("reasoning-label").textContent = models.thinking ? "推理已启用" : "推理未启用";
  renderSubtasks(state.subtasks);
  renderWorkflow(state.workflow);
  renderReact(state.react || []);
  renderLegend(state.scene_objects || []);
  const latestResult = [...(state.messages || [])].reverse().find(message => message.role === "assistant" && typeof message.success === "boolean");
  $("execution-note").textContent = state.agent === "tiptop"
    ? "开环执行模式：动作完成不等于目标已通过视觉验证。"
    : busy ? "实时显示执行阶段，完成后将进行视觉校验。"
      : latestResult ? TraceUI.completionLabel(latestResult) : "通过相机观测执行，以视觉校验结果。";
  if (loading) {
    $("camera-loading").hidden = false;
    $("frame-label").textContent = "等待场景就绪";
    $("elapsed").textContent = "—";
  }
  if (state.error && reportedError !== state.error) {
    reportedError = state.error;
    showDialog("需要处理", state.error);
  }
  if (!state.error) reportedError = null;
  renderMessages(state.messages || []);
  if (!loading) updateFrame(state.frame_id);
}

async function refreshState() {
  const epoch = stateEpoch;
  const sequence = ++readSequence;
  try {
    const response = await fetch("/api/state", { signal: AbortSignal.timeout(10000), cache: "no-store" });
    if (!response.ok) throw Error("服务不可用");
    const next = await response.json();
    if (epoch !== stateEpoch || sequence < appliedSequence) return;
    appliedSequence = sequence;
    connected = true;
    render(next);
    return next;
  } catch (_) {
    if (epoch !== stateEpoch || sequence < appliedSequence) return;
    appliedSequence = sequence;
    connected = false;
    $("connection").textContent = "连接断开，正在重连";
    $("connection-dot").style.background = "#c3965b";
    renderControls();
    $("scene-status").textContent = "离线";
    $("frame-label").textContent = "最后收到的画面";
  }
}

async function poll() {
  try { await refreshState(); }
  finally { setTimeout(poll, 350); }
}

$("command-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = $("instruction").value.trim();
  if (!text || !canMutate("command")) return;
  if (await mutate("command", "/api/command", { text })) $("instruction").value = "";
});
$("instruction").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("command-form").requestSubmit();
  }
});
document.querySelectorAll("[data-command]").forEach(button => button.addEventListener("click", () => {
  if (!canMutate("command")) return;
  $("instruction").value = button.dataset.command;
  $("instruction").focus();
}));
$("add-block").addEventListener("click", () => mutate("block", "/api/objects", { kind: "block" }));
$("add-box").addEventListener("click", () => mutate("box", "/api/objects", { kind: "box" }));
$("agent-select").addEventListener("change", async event => {
  const agent = event.target.value;
  if (!canMutate("agent") || agent === (state.agent || "pickparts")) {
    event.target.value = state?.agent || "pickparts";
    return;
  }
  await mutate("agent", "/api/agent", { agent });
  renderAgent();
});
// Scroll events also follow revealActiveStep and layout changes; only user intent pauses following.
["wheel", "touchmove"].forEach(type => $("workflow-lane").addEventListener(type, () => {
  followWorkflow = false;
}, { passive: true }));
$("workflow-lane").addEventListener("keydown", event => {
  if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) followWorkflow = false;
});
$("workflow-lane").addEventListener("pointerdown", event => {
  const lane = $("workflow-lane");
  const bounds = lane.getBoundingClientRect();
  if (event.clientX >= bounds.left + lane.clientLeft + lane.clientWidth
    || event.clientX < bounds.left + lane.clientLeft) followWorkflow = false;
}, { passive: true });
$("workflow-current").addEventListener("click", () => {
  followWorkflow = true;
  revealActiveStep();
  $("workflow-lane").focus({ preventScroll: true });
});
$("reset").addEventListener("click", () => {
  if (!canMutate("reset")) return;
  showDialog("随机重置场景？", "这会随机重新布局场景、复位机器人，并清空当前对话与执行记录。",
    () => mutate("reset", "/api/reset"));
});
poll();
