import { hasPlayableVideo, cacheBust } from "./mediaPreview.js";
import { isBusy, state } from "./workflowState.js";

export const els = {
  openingScreen: document.getElementById("openingScreen"),
  openingCanvas: document.getElementById("openingCanvas"),
  skipOpening: document.getElementById("skipOpening"),
  startCreating: document.getElementById("startCreating"),
  workflow: document.getElementById("workflow"),
  motionCard: document.getElementById("motionCard"),
  drawingCard: document.getElementById("drawingCard"),
  previewCard: document.getElementById("previewCard"),
  renderCard: document.getElementById("renderCard"),
  exportCard: document.getElementById("exportCard"),
  motionState: document.getElementById("motionState"),
  drawingState: document.getElementById("drawingState"),
  previewState: document.getElementById("previewState"),
  renderState: document.getElementById("renderState"),
  exportState: document.getElementById("exportState"),
  motionSelect: document.getElementById("motionSelect"),
  useMotion: document.getElementById("useMotion"),
  poseEstimator: document.getElementById("poseEstimator"),
  poseEstimatorWrap: document.getElementById("poseEstimatorWrap"),
  bvhUpload: document.getElementById("bvhUpload"),
  bvhFileName: document.getElementById("bvhFileName"),
  uploadBvh: document.getElementById("uploadBvh"),
  recordPreview: document.getElementById("recordPreview"),
  startRecord: document.getElementById("startRecord"),
  stopRecord: document.getElementById("stopRecord"),
  videoUpload: document.getElementById("videoUpload"),
  videoFileName: document.getElementById("videoFileName"),
  processVideo: document.getElementById("processVideo"),
  characterSelect: document.getElementById("characterSelect"),
  useCharacter: document.getElementById("useCharacter"),
  drawingUpload: document.getElementById("drawingUpload"),
  drawingFileName: document.getElementById("drawingFileName"),
  uploadDrawing: document.getElementById("uploadDrawing"),
  jointOverlay: document.getElementById("jointOverlay"),
  renderAnimation: document.getElementById("renderAnimation"),
  renderAgain: document.getElementById("renderAgain"),
  jobProgress: document.getElementById("jobProgress"),
  status: document.getElementById("status"),
  diagnosticsPanel: document.getElementById("diagnosticsPanel"),
  qualityWarnings: document.getElementById("qualityWarnings"),
  sourceVideo: document.getElementById("sourceVideo"),
  animationVideo: document.getElementById("animationVideo"),
  playBoth: document.getElementById("playBoth"),
  pauseBoth: document.getElementById("pauseBoth"),
  timeline: document.getElementById("timeline"),
  stepMotion: document.getElementById("stepMotion"),
  stepDrawing: document.getElementById("stepDrawing"),
  stepPreview: document.getElementById("stepPreview"),
  stepRender: document.getElementById("stepRender"),
  stepExport: document.getElementById("stepExport"),
  uploadLimits: document.getElementById("uploadLimits"),
  downloadAnimation: document.getElementById("downloadAnimation"),
  downloadBvh: document.getElementById("downloadBvh"),
  downloadPose: document.getElementById("downloadPose"),
  glassPoseCanvas: document.getElementById("glassPoseCanvas"),
  sampleButtons: Array.from(document.querySelectorAll(".sampleButton")),
};

export function setStatus(message) {
  state.statusMessage = message || "";
  els.status.textContent = state.statusMessage;
}

export function populateAssets(assets) {
  state.assets = assets;
  els.motionSelect.textContent = "";
  els.characterSelect.textContent = "";
  if (els.poseEstimator) els.poseEstimator.textContent = "";

  for (const motion of assets.motions) {
    const option = document.createElement("option");
    option.value = motion.id;
    option.dataset.retarget = motion.retarget_cfg;
    option.textContent = motion.name;
    els.motionSelect.appendChild(option);
  }

  for (const character of assets.characters) {
    const option = document.createElement("option");
    option.value = character.id;
    option.dataset.preview = character.preview_url || "";
    option.dataset.jointOverlay = character.joint_overlay_url || "";
    option.textContent = character.name;
    els.characterSelect.appendChild(option);
  }

  populatePoseEstimators(assets);
  updateSampleButtons();
  updateLimitText();
}

function populatePoseEstimators(assets) {
  if (!els.poseEstimator) return;
  const estimators = assets.pose_estimators || [];
  const visibleEstimators = estimators.filter((estimator) => estimator.id === "mediapipe" || estimator.available);
  for (const estimator of visibleEstimators) {
    const option = document.createElement("option");
    option.value = estimator.id;
    option.textContent = estimator.name;
    els.poseEstimator.appendChild(option);
  }
  els.poseEstimator.value = visibleEstimators.some((item) => item.id === assets.default_pose_estimator)
    ? assets.default_pose_estimator
    : "mediapipe";
  if (els.poseEstimatorWrap) {
    els.poseEstimatorWrap.hidden = visibleEstimators.length <= 1;
  }
}

export function updateSampleButtons() {
  for (const button of els.sampleButtons) {
    const sample = sampleById(button.dataset.sampleId);
    if (!sample) continue;
    button.querySelector("span").textContent = sample.label;
    button.querySelector("small").textContent = sample.description;
  }
}

export function updateLimitText() {
  const limits = state.assets?.limits;
  if (!limits) return;
  els.uploadLimits.textContent = `Video: ${limits.max_seconds}s, ${limits.max_video_width}x${limits.max_video_height}, ${limits.max_video_fps} FPS max. Uploads: ${limits.max_upload_mb} MB max.`;
}

export function sampleById(sampleId) {
  return (state.assets?.samples || []).find((sample) => sample.id === sampleId);
}

export function updateJobProgress(job) {
  els.jobProgress.hidden = false;
  els.jobProgress.value = Number(job.progress || 0);
}

export function clearJobProgress() {
  els.jobProgress.hidden = true;
  els.jobProgress.value = 0;
}

export function updateFileNames() {
  const videoFile = els.videoUpload.files[0];
  els.videoFileName.textContent = videoFile ? videoFile.name : state.recordedBlob ? "Recording ready" : "";
  const bvhFile = els.bvhUpload.files[0];
  els.bvhFileName.textContent = bvhFile ? bvhFile.name : "";
  const drawingFile = els.drawingUpload.files[0];
  els.drawingFileName.textContent = drawingFile ? drawingFile.name : "";
}

export function updateExportLinks() {
  setDownloadLink(els.downloadAnimation, state.animation?.animation_url);
  setDownloadLink(els.downloadBvh, state.motion?.bvh_url);
  setDownloadLink(els.downloadPose, state.motion?.pose_sequence_url);
}

export function renderDiagnostics() {
  if (!els.diagnosticsPanel || !state.diagnostics) return;
  const problemChecks = (state.diagnostics.checks || []).filter((check) => check.status !== "ok");
  if (!problemChecks.length) {
    els.diagnosticsPanel.hidden = true;
    els.diagnosticsPanel.textContent = "";
    return;
  }
  els.diagnosticsPanel.hidden = false;
  els.diagnosticsPanel.innerHTML = problemChecks
    .map((check) => `<p><strong>${escapeHtml(check.label)}</strong>: ${escapeHtml(check.message)}</p>`)
    .join("");
}

export function renderQualityWarnings() {
  if (!els.qualityWarnings) return;
  const warnings = state.motion?.quality_report?.warnings || [];
  if (!warnings.length) {
    els.qualityWarnings.hidden = true;
    els.qualityWarnings.textContent = "";
    return;
  }
  els.qualityWarnings.hidden = false;
  els.qualityWarnings.innerHTML = warnings.map((warning) => `<p>${escapeHtml(warning)}</p>`).join("");
}

export function renderUi() {
  const busy = isBusy();
  const recording = state.phase === "recording";
  const hasVideoInput = Boolean(els.videoUpload.files[0] || state.recordedBlob);
  const hasAssets = Boolean(state.assets);

  els.status.textContent = state.statusMessage || "";
  els.motionSelect.disabled = busy || recording || !hasAssets;
  els.useMotion.disabled = busy || recording || !els.motionSelect.options.length;
  if (els.poseEstimator) els.poseEstimator.disabled = busy || recording;
  els.bvhUpload.disabled = busy || recording;
  els.uploadBvh.disabled = busy || recording || !els.bvhUpload.files[0];
  els.startRecord.disabled = busy || recording;
  els.stopRecord.disabled = !recording;
  els.videoUpload.disabled = busy || recording;
  els.processVideo.disabled = busy || recording || !hasVideoInput;
  els.characterSelect.disabled = busy || recording || !hasAssets;
  els.useCharacter.disabled = busy || recording || !els.characterSelect.options.length;
  els.drawingUpload.disabled = busy || recording;
  els.uploadDrawing.disabled = busy || recording || !els.drawingUpload.files[0];
  els.renderAnimation.disabled = busy || recording || !state.motion || !state.drawing;
  els.renderAgain.disabled = busy || recording || !state.motion || !state.drawing;
  els.playBoth.disabled = !hasPlayableVideo(els.sourceVideo) && !hasPlayableVideo(els.animationVideo);
  els.pauseBoth.disabled = els.playBoth.disabled;
  for (const button of els.sampleButtons) {
    button.disabled = busy || recording || !hasAssets;
  }

  updateExportLinks();
  renderDiagnostics();
  renderQualityWarnings();
  renderSteps();
  renderPrimaryStep();
}

function setDownloadLink(link, url) {
  if (url) {
    link.href = cacheBust(url);
    link.hidden = false;
  } else {
    link.href = "#";
    link.hidden = true;
  }
}

function renderSteps() {
  const motionStatus = stepStatus("motion");
  const drawingStatus = stepStatus("drawing");
  const previewStatus = stepStatus("preview");
  const renderStatus = stepStatus("render");
  const exportStatus = stepStatus("export");

  setStep(els.stepMotion, motionStatus);
  setStep(els.stepDrawing, drawingStatus);
  setStep(els.stepPreview, previewStatus);
  setStep(els.stepRender, renderStatus);
  setStep(els.stepExport, exportStatus);

  setPill(els.motionState, motionStatus);
  setPill(els.drawingState, drawingStatus);
  setPill(els.previewState, previewStatus);
  setPill(els.renderState, renderStatus);
  setPill(els.exportState, exportStatus);
}

function stepStatus(step) {
  if (state.failedStep === step) return "failed";
  if (step === "motion") {
    if (state.phase === "motionProcessing" || state.phase === "recording") return "processing";
    if (state.motion) return "complete";
    return "ready";
  }
  if (step === "drawing") {
    if (state.phase === "drawingProcessing") return "processing";
    if (state.drawing) return "complete";
    return state.motion ? "ready" : "empty";
  }
  if (step === "preview") {
    if (state.motion && state.drawing) return "complete";
    return state.motion || state.drawing ? "ready" : "empty";
  }
  if (step === "render") {
    if (state.phase === "rendering") return "processing";
    if (state.animationReady) return "complete";
    return state.motion && state.drawing ? "ready" : "empty";
  }
  if (step === "export") {
    return state.animationReady ? "complete" : "empty";
  }
  return "empty";
}

function setStep(element, status) {
  setStatusClass(element, status);
}

function setPill(element, status) {
  setStatusClass(element, status);
  element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function setStatusClass(element, status) {
  element.classList.remove("empty", "ready", "processing", "complete", "failed");
  element.classList.add(status);
}

function renderPrimaryStep() {
  const primary = primaryCard();
  for (const card of [els.motionCard, els.drawingCard, els.previewCard, els.renderCard, els.exportCard]) {
    card.classList.toggle("primaryStep", card === primary);
  }
}

function primaryCard() {
  if (!state.motion || state.phase === "motionProcessing" || state.phase === "recording") return els.motionCard;
  if (!state.drawing || state.phase === "drawingProcessing") return els.drawingCard;
  if (!state.animationReady || state.phase === "rendering") return els.renderCard;
  return els.exportCard;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value || "";
  return div.innerHTML;
}
