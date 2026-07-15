const form = document.querySelector("#search-form");
const input = document.querySelector("#query");
const output = document.querySelector("#search-output");
const statusBox = document.querySelector("#status");
const resultsBox = document.querySelector("#results");
const button = form.querySelector("button");
const streamResults = document.querySelector("#stream-results");
let activeController = null;

function status(text, error = false) {
  output.hidden = false;
  statusBox.textContent = text;
  statusBox.classList.toggle("error", error);
}

function renderResults(data, complete) {
  resultsBox.replaceChildren();
  for (const item of data.results) {
    const row = document.createElement("div");
    row.className = "result";
    const text = document.createElement("span");
    text.textContent = item.text;
    const score = document.createElement("small");
    score.textContent = `频次 ${item.score}`;
    row.append(text, score);
    resultsBox.append(row);
  }
  const stopMessages = {
    node_limit: "达到节点检查上限",
    state_limit: "达到查询状态上限"
  };
  const stop = data.stop_reason ? stopMessages[data.stop_reason] || "搜索提前停止" : "";
  const tail = stop ? `；${stop}` : "";
  if (complete) {
    status(`找到 ${data.results.length} 条结果，检查 ${data.visited} 个索引节点${tail}。`);
  } else {
    status(`已找到 ${data.results.length} 条暂定结果，检查 ${data.visited} 个索引节点；仍在搜索……`);
  }
}

async function responseError(response) {
  const text = await response.text();
  return responseErrorFromText(response, text);
}

function responseErrorFromText(response, text) {
  if (text) {
    try {
      const data = JSON.parse(text);
      if (typeof data.error === "string") return data.error;
    } catch {
      const message = text.trim();
      if (message) return message;
    }
  }
  return `服务器返回 HTTP ${response.status}`;
}

function parseStreamEvent(line) {
  try {
    return JSON.parse(line);
  } catch {
    throw new Error("服务器返回了损坏的流式数据");
  }
}

async function runStreamingSearch(query, signal, showProgress) {
  const response = await fetch(`/api/search/stream?q=${encodeURIComponent(query)}&limit=100`, { signal });
  if (!response.ok) throw new Error(await responseError(response));
  if (!response.body) throw new Error("浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = parseStreamEvent(line);
      if (event.type === "error") throw new Error(event.error || "搜索失败");
      if (event.type === "queued") {
        status(`正在排队，前面还有 ${event.ahead} 个搜索任务……`);
      }
      if (event.type === "progress" && showProgress) renderResults(event, false);
      if (event.type === "complete") {
        renderResults(event, true);
        completed = true;
      }
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const event = parseStreamEvent(buffer);
    if (event.type === "error") throw new Error(event.error || "搜索失败");
    if (event.type === "queued") {
      status(`正在排队，前面还有 ${event.ahead} 个搜索任务……`);
    }
    if (event.type === "complete") {
      renderResults(event, true);
      completed = true;
    }
  }
  if (!completed) throw new Error("搜索连接提前结束");
}

async function runSearch(query) {
  if (activeController) activeController.abort();
  const controller = new AbortController();
  activeController = controller;
  button.disabled = true;
  resultsBox.replaceChildren();
  status("正在搜索……");
  try {
    await runStreamingSearch(query, controller.signal, streamResults.checked);
  } catch (error) {
    if (error.name !== "AbortError") {
      const message = error instanceof Error ? error.message : String(error);
      status(`搜索失败：${message}`, true);
    }
  } finally {
    if (activeController === controller) {
      activeController = null;
      button.disabled = false;
    }
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return status("请输入查询模式。", true);
  const url = new URL(location.href);
  url.searchParams.set("q", query);
  history.replaceState(null, "", url);
  runSearch(query);
});

const initial = new URL(location.href).searchParams.get("q");
if (initial) {
  input.value = initial;
  runSearch(initial);
}
