async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await response.json().catch(() => {
    return {
      success: false,
      error: "서버 응답을 JSON으로 해석하지 못했습니다.",
    };
  });

  return data;
}

export async function sendChatMessage(question, sessionId) {
  return requestJSON("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: sessionId,
    }),
  });
}

export async function getMemoryHistory(sessionId) {
  const params = new URLSearchParams({
    session_id: sessionId,
  });

  return requestJSON(`/api/memory/history?${params.toString()}`, {
    method: "GET",
  });
}

export async function resetMemory(sessionId) {
  return requestJSON("/api/memory/reset", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });
}

export async function resetMiddleware(sessionId) {
  return requestJSON("/api/middleware/reset", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });
}

export async function getMiddlewareLogs(sessionId, limit = 20) {
  const params = new URLSearchParams({
    session_id: sessionId,
    limit: String(limit),
  });

  return requestJSON(`/api/middleware/logs?${params.toString()}`, {
    method: "GET",
  });
}