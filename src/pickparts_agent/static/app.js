const $ = id => document.getElementById(id);
const ATOM_LABEL = {
  find_object: "定位", reach_above: "移动到上方", grasp: "抓取",
  lift: "抬起", carry_to: "搬运", release_into: "放入",
  verify_state: "校验", reset_arm: "复位", set_gripper: "夹爪",
};
let state = null, lastMessages = "", lastFrame = -1, started = null, frameBusy = false;
let reportedError = null, pending = false;
const context = $("camera").getContext("2d");

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
  });
  const result = await response.json();
  if (!response.ok) throw Error(typeof result.detail === "string" ? result.detail : "指令格式有误，请修改后重试。");
  return result;
}

function renderMessages(messages) {
  const encoded = JSON.stringify(messages);
  if (encoded === lastMessages) return;
  lastMessages = encoded;
  $("welcome").hidden = messages.length > 0;
  $("messages").replaceChildren();
  for (const message of messages) {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.textContent = message.role === "user" ? "你" : "PICKPARTS AGENT";
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = message.text;
    article.append(role, content);
    if (message.role === "assistant") {
      const badge = document.createElement("span");
      badge.className = "result-tag" + (message.success ? "" : " failed");
      badge.textContent = message.success ? "✓ 视觉确认完成" : "未完成 · 请查看结果";
      article.append(badge);
    }
    $("messages").append(article);
  }
  $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
}

function renderWorkflow(wf) {
  const lane = $("workflow-lane");
  lane.replaceChildren();
  if (!wf) {
    lane.textContent = "等待可执行的工作流…";
    return;
  }
  for (const step of wf.steps) {
    const node = document.createElement("div");
    node.className = `wf-node ${step.status}`;
    const idx = document.createElement("b");
    idx.textContent = String(step.index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = ATOM_LABEL[step.atom] || step.atom;
    node.append(idx, label);
    lane.append(node);
  }
}

function renderReact(entries) {
  const lane = $("react-lane");
  lane.replaceChildren();
  for (const entry of entries || []) {
    const item = document.createElement("div");
    item.className = "react-item";
    const tag = document.createElement("span");
    tag.className = "react-tag";
    tag.textContent = `第 ${entry.attempt} 次 · ${entry.decision}`;
    const text = document.createElement("p");
    text.textContent = `${entry.thought} ${entry.detail}`;
    item.append(tag, text);
    lane.append(item);
  }
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
  state = next;
  const busy = state.status === "busy";
  const loading = ["starting", "resetting"].includes(state.status);
  const ready = state.status === "ready" && !state.recovery_required;
  if (busy && !wasBusy) started = Date.now();
  if (!busy && wasBusy && started) $("elapsed").textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`;
  if (busy && started) $("elapsed").textContent = `${Math.floor((Date.now() - started) / 1000)} s`;
  $("connection").textContent = "本地服务已连接";
  $("connection-dot").style.background = "#6aaf70";
  $("scene-status").textContent = loading ? "初始化中" : busy ? "执行中" : state.status === "error" ? "需要处理" : "已连接";
  $("stage-label").textContent = state.stage;
  $("instruction").disabled = !ready || pending;
  $("send").disabled = !ready || pending;
  $("reset").disabled = loading || busy || pending;
  document.querySelectorAll("[data-command]").forEach(button => { button.disabled = !ready; });
  $("thinking").hidden = !busy;
  $("thinking-text").textContent = `${state.stage}…`;
  $("notice").hidden = !(state.error || state.recovery_required);
  $("notice").textContent = state.error || (state.recovery_required ? "操作中断，需重置场景后才能继续。" : "");
  $("model-label").textContent = state.models.llm
    ? `${state.models.llm} · 云端视觉 ${state.models.vision.replace(/^ark\//, "")}`
    : "云端模型等待就绪";
  renderWorkflow(state.workflow);
  renderReact(state.react);
  $("execution-note").textContent = state.stage === "已完成"
    ? "已通过 RGB-D 视觉校验。可继续发送下一条指令。"
    : busy ? "实时显示执行阶段，完成后将进行视觉校验。" : "通过相机观测执行，以视觉校验结果。";
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
  renderMessages(state.messages);
  if (!loading) updateFrame(state.frame_id);
}

async function poll() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) throw Error("服务不可用");
    render(await response.json());
  } catch (_) {
    $("connection").textContent = "连接断开，正在重连";
    $("connection-dot").style.background = "#c3965b";
    $("send").disabled = $("instruction").disabled = $("reset").disabled = true;
    $("scene-status").textContent = "离线";
    $("frame-label").textContent = "最后收到的画面";
  } finally { setTimeout(poll, 350); }
}

$("command-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = $("instruction").value.trim();
  if (!text || pending || state?.status !== "ready" || state.recovery_required) return;
  pending = true;
  $("send").disabled = true;
  try {
    await post("/api/command", { text });
    $("instruction").value = "";
  } catch (error) { showDialog("指令未发送", error.message); }
  finally { pending = false; }
});
$("instruction").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("command-form").requestSubmit();
  }
});
document.querySelectorAll("[data-command]").forEach(button => button.addEventListener("click", () => {
  $("instruction").value = button.dataset.command;
  $("instruction").focus();
}));
$("reset").addEventListener("click", () => showDialog("重新开始一次协作？",
  "这会恢复零件初始位置、机器人姿态，并清空当前对话。", async () => {
    pending = true;
    try {
      await post("/api/reset");
      lastFrame = -1;
    } catch (error) { showDialog("重置失败", error.message); }
    finally { pending = false; }
  }));
poll();
