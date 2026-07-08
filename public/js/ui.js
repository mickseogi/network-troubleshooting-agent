function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollChatToBottom() {
  const container = document.querySelector("#chat-messages");

  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

function getFlowClass(nodeName) {
  const name = nodeName.toLowerCase();

  if (name.includes("memory")) return "memory";
  if (name.includes("diagnose") || name.includes("rag") || name.includes("command")) return "tool";
  if (name.includes("conditional") || name.includes("clarification")) return "branch";
  if (name === "end") return "end";

  return "";
}

function normalizeFlowName(nodeName) {
  const map = {
    START: "START",
    END: "END",
    memory_node: "Memory",
    diagnose_node: "Diagnosis",
    rag_node: "RAG",
    command_node: "Command",
    generate_answer_node: "Answer",
    save_memory_node: "Save Memory",
    "conditional_edge:known_problem": "Known Problem",
    "conditional_edge:clarification": "Clarification",
    clarification_node: "Ask More",
  };

  return map[nodeName] || nodeName;
}

function splitRagDocuments(ragText) {
  if (!ragText) return [];

  const parts = ragText
    .split(/\n\n(?=\[문서\s+\d+\])/g)
    .map((item) => item.trim())
    .filter(Boolean);

  return parts.map((part) => {
    const sourceMatch = part.match(/source:\s*(.+)/);
    const source = sourceMatch ? sourceMatch[1].trim() : "unknown";

    const content = part
      .replace(/\[문서\s+\d+\]\s*source:\s*.+\n?/, "")
      .trim();

    return {
      source,
      content,
    };
  });
}

export function setLoading(isLoading) {
  const sendBtn = document.querySelector("#send-btn");
  const loadSessionBtn = document.querySelector("#load-session-btn");

  if (sendBtn) {
    sendBtn.disabled = isLoading;
    sendBtn.textContent = isLoading ? "Running..." : "Run Agent";
  }

  if (loadSessionBtn) {
    loadSessionBtn.disabled = isLoading;
  }

  if (isLoading) {
    setStatus("loading", "Running");
  }
}

export function setStatus(type, text) {
  const statusPill = document.querySelector("#status-pill");

  if (!statusPill) return;

  statusPill.className = `status-pill ${type}`;
  statusPill.textContent = text;
}

export function addUserMessage(message) {
  const container = document.querySelector("#chat-messages");

  if (!container) return;

  const element = document.createElement("div");
  element.className = "chat-message user";
  element.innerHTML = `
    <div class="avatar">ME</div>
    <div class="bubble">${escapeHTML(message)}</div>
  `;

  container.appendChild(element);
  scrollChatToBottom();
}

export function addAssistantMessage(message) {
  const container = document.querySelector("#chat-messages");

  if (!container) return;

  const element = document.createElement("div");
  element.className = "chat-message assistant";
  element.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">${escapeHTML(message || "응답이 없습니다.")}</div>
  `;

  container.appendChild(element);
  scrollChatToBottom();
}

export function showTypingMessage() {
  const container = document.querySelector("#chat-messages");

  if (!container) return;

  removeTypingMessage();

  const element = document.createElement("div");
  element.className = "chat-message assistant";
  element.id = "typing-message";
  element.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble typing-bubble"><span class="typing-indicator"><span></span><span></span><span></span></span></div>
  `;

  container.appendChild(element);
  scrollChatToBottom();
}

export function removeTypingMessage() {
  const typing = document.querySelector("#typing-message");

  if (typing) {
    typing.remove();
  }
}

export function resetChatMessages() {
  const container = document.querySelector("#chat-messages");

  if (!container) return;

  container.innerHTML = `
    <div class="chat-message assistant">
      <div class="avatar">AI</div>
      <div class="bubble compact-bubble">안녕하세요. 네트워크 장애 상황을 알려주세요.</div>
    </div>
  `;
}
export function renderChatMessagesFromHistory(history) {
  const container = document.querySelector("#chat-messages");

  if (!container) return;

  if (!history || history.length === 0) {
    resetChatMessages();
    return;
  }

  container.innerHTML = history
    .map((message) => {
      const isUser = message.role === "human";
      const roleClass = isUser ? "user" : "assistant";
      const avatar = isUser ? "ME" : "AI";

      return `
        <div class="chat-message ${roleClass}">
          <div class="avatar">${avatar}</div>
          <div class="bubble">${escapeHTML(message.content || "")}</div>
        </div>
      `;
    })
    .join("");

  scrollChatToBottom();
}

export function renderAgentResult(data) {
  const structured = data.structured_result || {};
  const stats = data.middleware_stats || {};

  const problemType = document.querySelector("#problem-type");
  const memorySaved = document.querySelector("#memory-saved");
  const modelCount = document.querySelector("#model-count");
  const toolCount = document.querySelector("#tool-count");
  const ragDocCount = document.querySelector("#rag-doc-count");

  setStatus(data.success ? "success" : "error", data.success ? "Success" : "Error");

  if (problemType) {
    problemType.textContent =
      data.diagnosis_tool_result || structured.problem_type || "-";
  }

  if (memorySaved) {
    memorySaved.textContent =
      typeof data.memory_saved === "boolean" ? String(data.memory_saved) : "-";
  }

  if (modelCount) {
    modelCount.textContent =
      stats.model_call_count !== undefined ? stats.model_call_count : "-";
  }

  if (toolCount) {
    toolCount.textContent =
      stats.tool_call_count !== undefined ? stats.tool_call_count : "-";
  }

  if (ragDocCount) {
    ragDocCount.textContent =
      data.rag_document_count !== undefined ? data.rag_document_count : "-";
  }

  renderStructuredResult(structured);
  renderGraphFlow(data.graph_flow || []);
  renderRagResult(data.rag_result || "");
  renderMiddlewareLogs(data.middleware_logs || []);
  renderChatHistory(data.chat_history || []);
}

export function renderStructuredResult(structured) {
  const causes = document.querySelector("#possible-causes");
  const commands = document.querySelector("#recommended-commands");
  const nextQuestion = document.querySelector("#next-question");

  const possibleCauses = structured.possible_causes || [];
  const recommendedCommands = structured.recommended_commands || [];

  if (causes) {
    if (possibleCauses.length === 0) {
      causes.innerHTML = "<li>아직 결과가 없습니다.</li>";
    } else {
      causes.innerHTML = possibleCauses
        .map((cause) => `<li>${escapeHTML(cause)}</li>`)
        .join("");
    }
  }

  if (commands) {
    if (recommendedCommands.length === 0) {
      commands.innerHTML = "<code>$ waiting for agent...</code>";
    } else {
      commands.innerHTML = recommendedCommands
        .map((command) => `<code>$ ${escapeHTML(command)}</code>`)
        .join("");
    }
  }

  if (nextQuestion) {
    nextQuestion.textContent = structured.next_question || "아직 결과가 없습니다.";
  }
}

export function renderGraphFlow(graphFlow) {
  const container = document.querySelector("#graph-flow");

  if (!container) return;

  if (!graphFlow.length) {
    container.innerHTML = `
      <span class="flow-chip muted">START</span>
      <span class="flow-arrow">→</span>
      <span class="flow-chip muted">Waiting</span>
    `;
    return;
  }

  container.innerHTML = graphFlow
    .map((node, index) => {
      const arrow = index < graphFlow.length - 1
        ? `<span class="flow-arrow">→</span>`
        : "";

      return `
        <span class="flow-chip ${getFlowClass(node)}">
          ${escapeHTML(normalizeFlowName(node))}
        </span>
        ${arrow}
      `;
    })
    .join("");
}

export function renderRagResult(ragText) {
  const container = document.querySelector("#rag-result");

  if (!container) return;

  if (!ragText) {
    container.textContent = "아직 검색 결과가 없습니다.";
    return;
  }

  const documents = splitRagDocuments(ragText);

  if (!documents.length) {
    container.innerHTML = `<pre>${escapeHTML(ragText)}</pre>`;
    return;
  }

  container.innerHTML = documents
    .map((doc) => {
      return `
        <article class="rag-doc">
          <span class="rag-source">${escapeHTML(doc.source)}</span>
          <pre>${escapeHTML(doc.content)}</pre>
        </article>
      `;
    })
    .join("");
}

export function renderMiddlewareLogs(logs) {
  const container = document.querySelector("#middleware-logs");

  if (!container) return;

  if (!logs.length) {
    container.textContent = "아직 middleware 로그가 없습니다.";
    return;
  }

  container.innerHTML = logs
    .slice()
    .reverse()
    .map((log) => {
      const level = log.level || "info";

      return `
        <article class="log-item">
          <div class="log-top">
            <span class="log-level ${escapeHTML(level)}">${escapeHTML(level)}</span>
            <span class="log-middleware">${escapeHTML(log.middleware || "-")}</span>
            <span class="log-time">${escapeHTML(log.timestamp || "")}</span>
          </div>
          <div class="log-message">${escapeHTML(log.message || "")}</div>
          ${
            log.detail
              ? `<p class="log-detail">${escapeHTML(log.detail)}</p>`
              : ""
          }
        </article>
      `;
    })
    .join("");
}

export function renderChatHistory(history) {
  const container = document.querySelector("#chat-history");

  if (!container) return;

  if (!history.length) {
    container.textContent = "아직 대화 이력이 없습니다.";
    return;
  }

  container.innerHTML = history
    .map((message) => {
      const role = message.role === "human" ? "User" : "AI";

      return `
        <article class="history-item">
          <span class="history-role">${escapeHTML(role)}</span>
          <p class="history-content">${escapeHTML(message.content || "")}</p>
        </article>
      `;
    })
    .join("");
}

export function resetAnalysisView() {
  const problemType = document.querySelector("#problem-type");
  const memorySaved = document.querySelector("#memory-saved");
  const modelCount = document.querySelector("#model-count");
  const toolCount = document.querySelector("#tool-count");
  const ragDocCount = document.querySelector("#rag-doc-count");

  if (problemType) problemType.textContent = "-";
  if (memorySaved) memorySaved.textContent = "-";
  if (modelCount) modelCount.textContent = "-";
  if (toolCount) toolCount.textContent = "-";
  if (ragDocCount) ragDocCount.textContent = "-";

  renderStructuredResult({});
  renderGraphFlow([]);
  renderRagResult("");
  renderMiddlewareLogs([]);
  renderChatHistory([]);
}

export function activateTab(tabName) {
  const buttons = document.querySelectorAll(".tab-button");
  const panels = document.querySelectorAll(".tab-panel");

  buttons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });

  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}