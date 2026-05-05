import { state } from "./workflowState.js";

export function errorMessage(data, fallback) {
  if (data && data.error) {
    if (typeof data.error === "string") return data.error;
    if (data.error.message) return data.error.message;
  }
  return fallback || "Request failed";
}

export async function requestJson(url, options = {}) {
  const response = await fetch(url, { credentials: "same-origin", ...options });
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      throw new Error(text);
    }
  }
  if (!response.ok) {
    throw new Error(errorMessage(data, response.statusText));
  }
  return data;
}

export async function postForm(url, form) {
  form.append("session_id", state.sessionId);
  form.append("csrf_token", state.csrfToken);
  return requestJson(url, {
    method: "POST",
    headers: { "X-CSRF-Token": state.csrfToken },
    body: form,
  });
}

export async function postJson(url, payload) {
  return requestJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": state.csrfToken,
    },
    body: JSON.stringify({ ...payload, session_id: state.sessionId, csrf_token: state.csrfToken }),
  });
}
