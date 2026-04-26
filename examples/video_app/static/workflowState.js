const busyPhases = new Set(["motionProcessing", "drawingProcessing", "rendering"]);

export const state = {
  sessionId: window.APP_SESSION_ID || String(Date.now()),
  csrfToken: window.APP_CSRF_TOKEN || "",
  phase: "idle",
  statusMessage: "",
  failedStep: null,
  assets: null,
  diagnostics: null,
  recordedBlob: null,
  recordPreviewUrl: null,
  mediaRecorder: null,
  stream: null,
  motion: null,
  drawing: null,
  animation: null,
  animationReady: false,
  activeJob: null,
  pollTimer: null,
};

export function isBusy() {
  return busyPhases.has(state.phase);
}

export function setStatusMessage(message) {
  state.statusMessage = message || "";
}

export function beginPhase(phase, message) {
  state.failedStep = null;
  setPhase(phase, message);
}

export function setPhase(phase, message) {
  state.phase = phase;
  if (message !== undefined) setStatusMessage(message);
}

export function readyMessage(defaultMessage) {
  if (state.motion && state.drawing) return "Ready to render.";
  return defaultMessage;
}

export function markFailedStep(phase) {
  if (phase === "motionProcessing" || phase === "recording") state.failedStep = "motion";
  else if (phase === "drawingProcessing") state.failedStep = "drawing";
  else if (phase === "rendering") state.failedStep = "render";
  else state.failedStep = "preview";
}
