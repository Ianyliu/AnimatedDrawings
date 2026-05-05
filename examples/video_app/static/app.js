import { postForm, postJson, requestJson } from "./api.js";
import { startGlassPoseAnimation } from "./glassPose.js";
import { clearPollTimer, trackJob } from "./jobs.js";
import {
  clearVideo,
  describeVideoError,
  pauseBoth,
  playBoth,
  seekBoth,
  setImageSource,
  setVideoSource,
  startRecording,
  stopRecording,
  syncTimeline,
} from "./mediaPreview.js";
import { startOpeningExperience } from "./openingExperience.js";
import {
  clearJobProgress,
  els,
  populateAssets,
  renderUi,
  sampleById,
  setStatus,
  updateFileNames,
} from "./renderUi.js";
import { beginPhase, isBusy, markFailedStep, readyMessage, setPhase, state } from "./workflowState.js";

function setPhaseAndRender(phase, message) {
  setPhase(phase, message);
  renderUi();
}

function beginPhaseAndRender(phase, message) {
  beginPhase(phase, message);
  renderUi();
}

async function loadAssets() {
  const assets = await requestJson("/api/assets");
  populateAssets(assets);
  setStatus("Choose a motion or try a sample.");
  renderUi();
}

async function loadDiagnostics() {
  const diagnostics = await requestJson("/api/diagnostics");
  state.diagnostics = diagnostics;
  renderUi();
}

function selectedMotionPayload() {
  const selected = els.motionSelect.selectedOptions[0];
  if (!selected) return null;
  return {
    motion_cfg: selected.value,
    retarget_cfg: selected.dataset.retarget,
    overlay_url: null,
    bvh_url: null,
    pose_sequence_url: null,
    quality_report: null,
  };
}

function useBundledMotion() {
  if (isBusy()) return;
  const motion = selectedMotionPayload();
  if (!motion) return;
  applyMotion(motion);
  setPhaseAndRender(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Motion selected."));
}

async function useBundledCharacter() {
  if (isBusy()) return;
  const selected = els.characterSelect.selectedOptions[0];
  if (!selected) return;
  beginPhaseAndRender("drawingProcessing", "Loading character joints...");
  const data = await loadBundledCharacter(selected.value);
  applyDrawing(data);
  setPhaseAndRender(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Character ready."));
}

async function loadBundledCharacter(characterCfg) {
  const form = new FormData();
  form.append("character_cfg", characterCfg);
  return postForm("/api/drawing", form);
}

async function useSample(sampleId) {
  if (isBusy()) return;
  const sample = sampleById(sampleId);
  if (!sample) {
    setStatus("Sample is unavailable.");
    return;
  }

  beginPhaseAndRender("drawingProcessing", "Loading sample...");
  selectOptionByValue(els.motionSelect, sample.motion_cfg);
  selectOptionByValue(els.characterSelect, sample.character_cfg);
  applyMotion({
    motion_cfg: sample.motion_cfg,
    retarget_cfg: sample.retarget_cfg,
    overlay_url: null,
    bvh_url: null,
    pose_sequence_url: null,
    quality_report: null,
  });
  const drawing = await loadBundledCharacter(sample.character_cfg);
  applyDrawing(drawing);
  setPhaseAndRender("readyToRender", "Sample ready. Render when ready.");
}

function selectOptionByValue(select, value) {
  for (const option of select.options) {
    if (option.value === value) {
      select.value = value;
      return;
    }
  }
}

async function processVideo() {
  const form = new FormData();
  const uploaded = els.videoUpload.files[0];
  if (uploaded) {
    form.append("video", uploaded);
  } else if (state.recordedBlob) {
    form.append("video", state.recordedBlob, "recording.webm");
  } else {
    setStatus("Select or record a video first.");
    return;
  }

  form.append("max_seconds", String(window.APP_MAX_SECONDS));
  if (els.poseEstimator?.value) form.append("pose_estimator", els.poseEstimator.value);
  beginPhaseAndRender("motionProcessing", "Uploading video...");
  const data = await postForm("/api/motion/video", form);
  await trackJob(data.job, "motionProcessing", (result) => {
    applyMotion(result);
    setPhaseAndRender(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Video motion ready."));
  });
}

async function uploadBvh() {
  const file = els.bvhUpload.files[0];
  if (!file) {
    setStatus("Select a BVH file first.");
    return;
  }
  const form = new FormData();
  form.append("bvh", file);
  beginPhaseAndRender("motionProcessing", "Uploading BVH...");
  const data = await postForm("/api/motion/bvh", form);
  await trackJob(data.job, "motionProcessing", (result) => {
    applyMotion(result);
    setPhaseAndRender(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("BVH motion ready."));
  });
}

async function uploadDrawing() {
  const file = els.drawingUpload.files[0];
  if (!file) {
    setStatus("Select a drawing first.");
    return;
  }
  const form = new FormData();
  form.append("drawing", file);
  beginPhaseAndRender("drawingProcessing", "Uploading drawing...");
  const data = await postForm("/api/drawing", form);
  await trackJob(data.job, "drawingProcessing", (result) => {
    applyDrawing(result);
    setPhaseAndRender(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Drawing ready."));
  });
}

async function renderAnimation() {
  if (!state.motion || !state.drawing) {
    setStatus("Choose a motion and drawing first.");
    return;
  }
  beginPhaseAndRender("rendering", "Starting render...");
  const data = await postJson("/api/render", {
    character_cfg: state.drawing.character_cfg,
    motion_cfg: state.motion.motion_cfg,
    retarget_cfg: state.motion.retarget_cfg,
  });
  await trackJob(data.job, "rendering", (result) => {
    applyAnimation(result);
    setPhaseAndRender("complete", "Animation ready.");
  });
}

function applyMotion(data) {
  state.motion = data;
  if (data.overlay_url) {
    setVideoSource(els.sourceVideo, data.overlay_url, els);
  } else {
    clearVideo(els.sourceVideo, els);
  }
  invalidateAnimation();
}

function applyDrawing(data) {
  state.drawing = data;
  setImageSource(els.jointOverlay, data.joint_overlay_url);
  invalidateAnimation();
}

function applyAnimation(data) {
  state.animation = data;
  setVideoSource(els.animationVideo, data.animation_url, els);
  state.animationReady = true;
  renderUi();
}

function invalidateAnimation() {
  state.animationReady = false;
  state.animation = null;
  clearVideo(els.animationVideo, els);
  renderUi();
}

function handleError(error) {
  clearPollTimer();
  clearJobProgress();
  markFailedStep(state.phase);
  setPhaseAndRender("failed", error.message || "Request failed.");
}

function shouldFocusAfterScroll() {
  if (!window.matchMedia) return true;
  return !window.matchMedia("(max-width: 700px), (pointer: coarse)").matches;
}

function bindEvents() {
  els.startCreating.addEventListener("click", () => {
    els.workflow.scrollIntoView({ behavior: "smooth", block: "start" });
    if (shouldFocusAfterScroll()) {
      window.setTimeout(() => els.motionSelect.focus(), 300);
    }
  });
  els.useMotion.addEventListener("click", useBundledMotion);
  els.useCharacter.addEventListener("click", () => useBundledCharacter().catch(handleError));
  els.startRecord.addEventListener("click", () => {
    startRecording(els, beginPhaseAndRender, () => {
      updateFileNames();
      setPhaseAndRender("idle", "Recording ready.");
    }).catch(handleError);
  });
  els.stopRecord.addEventListener("click", stopRecording);
  els.videoUpload.addEventListener("change", () => {
    state.recordedBlob = null;
    updateFileNames();
    renderUi();
  });
  els.bvhUpload.addEventListener("change", () => {
    updateFileNames();
    renderUi();
  });
  els.drawingUpload.addEventListener("change", () => {
    updateFileNames();
    renderUi();
  });
  for (const button of els.sampleButtons) {
    button.addEventListener("click", () => useSample(button.dataset.sampleId).catch(handleError));
  }
  els.processVideo.addEventListener("click", () => processVideo().catch(handleError));
  els.uploadBvh.addEventListener("click", () => uploadBvh().catch(handleError));
  els.uploadDrawing.addEventListener("click", () => uploadDrawing().catch(handleError));
  els.renderAnimation.addEventListener("click", () => renderAnimation().catch(handleError));
  els.renderAgain.addEventListener("click", () => renderAnimation().catch(handleError));
  els.playBoth.addEventListener("click", () => playBoth(els, setStatus));
  els.pauseBoth.addEventListener("click", () => pauseBoth(els));
  els.timeline.addEventListener("input", (event) => seekBoth(els, event.target.value));
  els.animationVideo.addEventListener("loadedmetadata", () => {
    syncTimeline(els);
    renderUi();
  });
  els.sourceVideo.addEventListener("loadedmetadata", () => {
    syncTimeline(els);
    renderUi();
  });
  els.animationVideo.addEventListener("error", () =>
    setStatus(`Animation video failed to load: ${describeVideoError(els.animationVideo)}.`)
  );
  els.sourceVideo.addEventListener("error", () =>
    setStatus(`Source video failed to load: ${describeVideoError(els.sourceVideo)}.`)
  );
  els.animationVideo.addEventListener("timeupdate", () => {
    els.timeline.value = String(els.animationVideo.currentTime || 0);
    if (
      els.sourceVideo.currentSrc &&
      Math.abs(els.sourceVideo.currentTime - els.animationVideo.currentTime) > 0.2
    ) {
      els.sourceVideo.currentTime = Math.min(
        els.animationVideo.currentTime,
        els.sourceVideo.duration || els.animationVideo.currentTime
      );
    }
  });
}

bindEvents();
startGlassPoseAnimation(els.glassPoseCanvas);
updateFileNames();
renderUi();
startOpeningExperience({
  screen: els.openingScreen,
  canvas: els.openingCanvas,
  skipButton: els.skipOpening,
  maxDurationMs: 4600,
});
loadAssets().catch(handleError);
loadDiagnostics().catch(() => {});
