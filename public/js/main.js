import {
  sendChatMessage,
  resetMemory,
  resetMiddleware,
  getMemoryHistory,
} from "./api.js";

import {
  activateTab,
  addAssistantMessage,
  addUserMessage,
  renderAgentResult,
  renderChatMessagesFromHistory,
  renderMiddlewareLogs,
  removeTypingMessage,
  resetAnalysisView,
  resetChatMessages,
  setLoading,
  setStatus,
  showTypingMessage,
} from "./ui.js";

const form = document.querySelector("#chat-form");
const sessionInput = document.querySelector("#session-id");
const questionInput = document.querySelector("#question-input");
const loadSessionBtn = document.querySelector("#load-session-btn");
const memoryResetBtn = document.querySelector("#memory-reset-btn");
const middlewareResetBtn = document.querySelector("#middleware-reset-btn");
const sampleButtons = document.querySelectorAll(".sample-btn");
const tabButtons = document.querySelectorAll(".tab-button");

function getSessionId() {
  const value = sessionInput.value.trim();
  return value || "default";
}

function saveLastSession(sessionId) {
  localStorage.setItem("network-agent-last-session", sessionId);
}

function loadLastSessionId() {
  const savedSession = localStorage.getItem("network-agent-last-session");

  if (savedSession) {
    sessionInput.value = savedSession;
  }
}

async function loadSession(sessionId = getSessionId()) {
  sessionInput.value = sessionId;
  saveLastSession(sessionId);

  setStatus("loading", "Loading");
  loadSessionBtn.disabled = true;

  try {
    const data = await getMemoryHistory(sessionId);

    if (!data.success) {
      addAssistantMessage(data.error || "세션을 불러오지 못했습니다.");
      setStatus("error", "Error");
      return;
    }

    renderChatMessagesFromHistory(data.chat_history || []);
    resetAnalysisView();

    if (data.chat_history && data.chat_history.length > 0) {
      setStatus("success", "Session Loaded");
    } else {
      setStatus("success", "New Session");
    }
  } catch (error) {
    addAssistantMessage(error.message || "세션을 불러오는 중 오류가 발생했습니다.");
    setStatus("error", "Error");
  } finally {
    loadSessionBtn.disabled = false;
  }
}

async function handleSubmit(event) {
  event.preventDefault();

  const question = questionInput.value.trim();
  const sessionId = getSessionId();

  if (!question) {
    addAssistantMessage("질문을 입력해주세요.");
    setStatus("error", "Error");
    return;
  }

  saveLastSession(sessionId);

  addUserMessage(question);
  questionInput.value = "";
  setLoading(true);
  showTypingMessage();

  try {
    const data = await sendChatMessage(question, sessionId);

    removeTypingMessage();

    if (!data.success) {
      addAssistantMessage(`오류가 발생했습니다.\n${data.error || "알 수 없는 오류입니다."}`);
      renderAgentResult(data);
      return;
    }

    addAssistantMessage(data.answer || "응답이 없습니다.");
    renderAgentResult(data);
  } catch (error) {
    removeTypingMessage();
    addAssistantMessage(`오류가 발생했습니다.\n${error.message || "알 수 없는 오류입니다."}`);
    setStatus("error", "Error");
  } finally {
    setLoading(false);
  }
}

async function handleMemoryReset() {
  const sessionId = getSessionId();

  memoryResetBtn.disabled = true;

  try {
    const data = await resetMemory(sessionId);

    if (!data.success) {
      addAssistantMessage(data.error || "Memory 초기화에 실패했습니다.");
      setStatus("error", "Error");
      return;
    }

    resetChatMessages();
    resetAnalysisView();
    setStatus("success", "Memory Reset");
  } catch (error) {
    addAssistantMessage(error.message || "Memory 초기화 중 오류가 발생했습니다.");
    setStatus("error", "Error");
  } finally {
    memoryResetBtn.disabled = false;
  }
}

async function handleMiddlewareReset() {
  const sessionId = getSessionId();

  middlewareResetBtn.disabled = true;

  try {
    const data = await resetMiddleware(sessionId);

    if (!data.success) {
      addAssistantMessage(data.error || "Middleware 로그 초기화에 실패했습니다.");
      setStatus("error", "Error");
      return;
    }

    renderMiddlewareLogs([]);

    const modelCount = document.querySelector("#model-count");
    const toolCount = document.querySelector("#tool-count");

    if (modelCount) modelCount.textContent = "0";
    if (toolCount) toolCount.textContent = "0";

    setStatus("success", "Logs Reset");
  } catch (error) {
    addAssistantMessage(error.message || "Middleware 로그 초기화 중 오류가 발생했습니다.");
    setStatus("error", "Error");
  } finally {
    middlewareResetBtn.disabled = false;
  }
}

function bindSampleButtons() {
  sampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.textContent.trim();
      questionInput.focus();
    });
  });
}

function bindTabs() {
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activateTab(button.dataset.tab);
    });
  });
}

function bindEnterSubmit() {
  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
}

function bindEvents() {
  form.addEventListener("submit", handleSubmit);

  loadSessionBtn.addEventListener("click", () => {
    loadSession(getSessionId());
  });

  memoryResetBtn.addEventListener("click", handleMemoryReset);
  middlewareResetBtn.addEventListener("click", handleMiddlewareReset);

  bindSampleButtons();
  bindTabs();
  bindEnterSubmit();
}

function init() {
  loadLastSessionId();
  bindEvents();
}

init();