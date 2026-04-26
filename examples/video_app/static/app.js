const busyPhases = new Set(["motionProcessing", "drawingProcessing", "rendering"]);

const state = {
  sessionId: window.APP_SESSION_ID || String(Date.now()),
  phase: "idle",
  failedStep: null,
  assets: null,
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

const els = {
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
  sampleButtons: Array.from(document.querySelectorAll(".sampleButton")),
};

function isBusy() {
  return busyPhases.has(state.phase);
}

function hasPlayableVideo(video) {
  return Boolean(video.currentSrc || video.getAttribute("src"));
}

function setStatus(message) {
  els.status.textContent = message || "";
}

function beginPhase(phase, message) {
  state.failedStep = null;
  setPhase(phase, message);
}

function setPhase(phase, message) {
  state.phase = phase;
  if (message !== undefined) setStatus(message);
  renderUi();
}

function cacheBust(url) {
  if (!url) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}client_ts=${Date.now()}`;
}

function clearVideo(video) {
  video.pause();
  video.removeAttribute("src");
  video.load();
  syncTimeline();
}

function setVideoSource(video, url) {
  clearVideo(video);
  if (!url) return;
  video.src = cacheBust(url);
  video.load();
}

function setImageSource(image, url) {
  if (!url) {
    image.removeAttribute("src");
    return;
  }
  image.src = cacheBust(url);
}

function invalidateAnimation() {
  state.animationReady = false;
  state.animation = null;
  clearVideo(els.animationVideo);
  updateExportLinks();
}

function errorMessage(data, fallback) {
  if (data && data.error) {
    if (typeof data.error === "string") return data.error;
    if (data.error.message) return data.error.message;
  }
  return fallback || "Request failed";
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
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

async function postForm(url, form) {
  form.append("session_id", state.sessionId);
  return requestJson(url, { method: "POST", body: form });
}

async function postJson(url, payload) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, session_id: state.sessionId }),
  });
}

async function loadAssets() {
  const assets = await requestJson("/api/assets");
  state.assets = assets;

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

  updateSampleButtons();
  updateLimitText();
  setStatus("Choose a motion or try a sample.");
  renderUi();
}

function updateSampleButtons() {
  for (const button of els.sampleButtons) {
    const sample = sampleById(button.dataset.sampleId);
    if (!sample) continue;
    button.querySelector("span").textContent = sample.label;
    button.querySelector("small").textContent = sample.description;
  }
}

function updateLimitText() {
  const limits = state.assets?.limits;
  if (!limits) return;
  els.uploadLimits.textContent = `Video: ${limits.max_seconds}s, ${limits.max_video_width}x${limits.max_video_height}, ${limits.max_video_fps} FPS max. Uploads: ${limits.max_upload_mb} MB max.`;
}

function sampleById(sampleId) {
  return (state.assets?.samples || []).find((sample) => sample.id === sampleId);
}

function readyMessage(defaultMessage) {
  if (state.motion && state.drawing) return "Ready to render.";
  return defaultMessage;
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
  };
}

function useBundledMotion() {
  if (isBusy()) return;
  const motion = selectedMotionPayload();
  if (!motion) return;
  applyMotion(motion);
  setPhase(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Motion selected."));
}

async function useBundledCharacter() {
  if (isBusy()) return;
  const selected = els.characterSelect.selectedOptions[0];
  if (!selected) return;
  beginPhase("drawingProcessing", "Loading character joints...");
  const data = await loadBundledCharacter(selected.value);
  applyDrawing(data);
  setPhase(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Character ready."));
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

  beginPhase("drawingProcessing", "Loading sample...");
  selectOptionByValue(els.motionSelect, sample.motion_cfg);
  selectOptionByValue(els.characterSelect, sample.character_cfg);
  applyMotion({
    motion_cfg: sample.motion_cfg,
    retarget_cfg: sample.retarget_cfg,
    overlay_url: null,
    bvh_url: null,
    pose_sequence_url: null,
  });
  const drawing = await loadBundledCharacter(sample.character_cfg);
  applyDrawing(drawing);
  setPhase("readyToRender", "Sample ready. Render when ready.");
}

function selectOptionByValue(select, value) {
  for (const option of select.options) {
    if (option.value === value) {
      select.value = value;
      return;
    }
  }
}

async function startRecording() {
  if (isBusy()) return;
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  els.recordPreview.srcObject = state.stream;
  await els.recordPreview.play();

  const chunks = [];
  state.mediaRecorder = new MediaRecorder(state.stream);
  state.mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  state.mediaRecorder.onstop = () => {
    state.recordedBlob = new Blob(chunks, { type: state.mediaRecorder.mimeType || "video/webm" });
    if (state.recordPreviewUrl) URL.revokeObjectURL(state.recordPreviewUrl);
    state.recordPreviewUrl = URL.createObjectURL(state.recordedBlob);
    els.recordPreview.srcObject = null;
    els.recordPreview.src = state.recordPreviewUrl;
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    updateFileNames();
    setPhase("idle", "Recording ready.");
  };

  state.mediaRecorder.start();
  beginPhase("recording", `Recording up to ${window.APP_MAX_SECONDS}s...`);
  setTimeout(() => {
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
    }
  }, window.APP_MAX_SECONDS * 1000);
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
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
  beginPhase("motionProcessing", "Uploading video...");
  const data = await postForm("/api/motion/video", form);
  await trackJob(data.job, "motionProcessing", (result) => {
    applyMotion(result);
    setPhase(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Video motion ready."));
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
  beginPhase("motionProcessing", "Uploading BVH...");
  const data = await postForm("/api/motion/bvh", form);
  await trackJob(data.job, "motionProcessing", (result) => {
    applyMotion(result);
    setPhase(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("BVH motion ready."));
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
  beginPhase("drawingProcessing", "Uploading drawing...");
  const data = await postForm("/api/drawing", form);
  await trackJob(data.job, "drawingProcessing", (result) => {
    applyDrawing(result);
    setPhase(state.motion && state.drawing ? "readyToRender" : "idle", readyMessage("Drawing ready."));
  });
}

async function renderAnimation() {
  if (!state.motion || !state.drawing) {
    setStatus("Choose a motion and drawing first.");
    return;
  }
  beginPhase("rendering", "Starting render...");
  const data = await postJson("/api/render", {
    character_cfg: state.drawing.character_cfg,
    motion_cfg: state.motion.motion_cfg,
    retarget_cfg: state.motion.retarget_cfg,
  });
  await trackJob(data.job, "rendering", (result) => {
    applyAnimation(result);
    setPhase("complete", "Animation ready.");
  });
}

function applyMotion(data) {
  state.motion = data;
  if (data.overlay_url) {
    setVideoSource(els.sourceVideo, data.overlay_url);
  } else {
    clearVideo(els.sourceVideo);
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
  setVideoSource(els.animationVideo, data.animation_url);
  state.animationReady = true;
  updateExportLinks();
}

async function trackJob(job, phase, onComplete) {
  if (!job) throw new Error("Server did not return a job.");
  clearPollTimer();
  state.activeJob = job;
  setPhase(phase, job.message || "Queued.");

  return new Promise((resolve, reject) => {
    const consume = (nextJob) => {
      state.activeJob = nextJob;
      updateJobProgress(nextJob);
      setStatus(nextJob.message || "");
      renderUi();

      if (nextJob.status === "completed") {
        clearPollTimer();
        state.activeJob = null;
        clearJobProgress();
        onComplete(nextJob.result || {});
        resolve(nextJob.result || {});
        return;
      }

      if (nextJob.status === "failed") {
        clearPollTimer();
        state.activeJob = null;
        clearJobProgress();
        const message = nextJob.error?.message || "Job failed.";
        markFailedStep(phase);
        setPhase("failed", message);
        reject(new Error(message));
        return;
      }

      state.pollTimer = setTimeout(async () => {
        try {
          const data = await requestJson(nextJob.status_url);
          consume(data.job);
        } catch (e) {
          clearPollTimer();
          state.activeJob = null;
          clearJobProgress();
          markFailedStep(phase);
          setPhase("failed", e.message);
          reject(e);
        }
      }, 700);
    };

    consume(job);
  });
}

function markFailedStep(phase) {
  if (phase === "motionProcessing" || phase === "recording") state.failedStep = "motion";
  else if (phase === "drawingProcessing") state.failedStep = "drawing";
  else if (phase === "rendering") state.failedStep = "render";
  else state.failedStep = "preview";
}

function clearPollTimer() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function updateJobProgress(job) {
  els.jobProgress.hidden = false;
  els.jobProgress.value = Number(job.progress || 0);
}

function clearJobProgress() {
  els.jobProgress.hidden = true;
  els.jobProgress.value = 0;
}

function updateFileNames() {
  const videoFile = els.videoUpload.files[0];
  els.videoFileName.textContent = videoFile ? videoFile.name : state.recordedBlob ? "Recording ready" : "";
  const bvhFile = els.bvhUpload.files[0];
  els.bvhFileName.textContent = bvhFile ? bvhFile.name : "";
  const drawingFile = els.drawingUpload.files[0];
  els.drawingFileName.textContent = drawingFile ? drawingFile.name : "";
}

function updateExportLinks() {
  setDownloadLink(els.downloadAnimation, state.animation?.animation_url);
  setDownloadLink(els.downloadBvh, state.motion?.bvh_url);
  setDownloadLink(els.downloadPose, state.motion?.pose_sequence_url);
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

function renderUi() {
  const busy = isBusy();
  const recording = state.phase === "recording";
  const hasVideoInput = Boolean(els.videoUpload.files[0] || state.recordedBlob);
  const hasAssets = Boolean(state.assets);

  els.motionSelect.disabled = busy || recording || !hasAssets;
  els.useMotion.disabled = busy || recording || !els.motionSelect.options.length;
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
  renderSteps();
  renderPrimaryStep();
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

function handleError(error) {
  clearPollTimer();
  clearJobProgress();
  markFailedStep(state.phase);
  setPhase("failed", error.message || "Request failed.");
}

function syncTimeline() {
  const durations = [els.animationVideo.duration, els.sourceVideo.duration].filter(Number.isFinite);
  const duration = durations.length ? Math.max(...durations) : 0;
  els.timeline.max = String(duration || 0);
}

function playBoth() {
  if (hasPlayableVideo(els.sourceVideo)) els.sourceVideo.play().catch((e) => setStatus(e.message));
  if (hasPlayableVideo(els.animationVideo)) els.animationVideo.play().catch((e) => setStatus(e.message));
}

function pauseBoth() {
  els.sourceVideo.pause();
  els.animationVideo.pause();
}

function seekBoth(value) {
  const time = Number(value);
  if (hasPlayableVideo(els.sourceVideo)) els.sourceVideo.currentTime = Math.min(time, els.sourceVideo.duration || time);
  if (hasPlayableVideo(els.animationVideo)) els.animationVideo.currentTime = Math.min(time, els.animationVideo.duration || time);
}

function describeVideoError(video) {
  if (!video.error) return "unknown error";
  const messages = {
    1: "loading was aborted",
    2: "network error",
    3: "decode error",
    4: "source not supported",
  };
  return messages[video.error.code] || `error code ${video.error.code}`;
}

function shouldFocusAfterScroll() {
  if (!window.matchMedia) return true;
  return !window.matchMedia("(max-width: 700px), (pointer: coarse)").matches;
}

els.startCreating.addEventListener("click", () => {
  els.workflow.scrollIntoView({ behavior: "smooth", block: "start" });
  if (shouldFocusAfterScroll()) {
    window.setTimeout(() => els.motionSelect.focus(), 300);
  }
});
els.useMotion.addEventListener("click", useBundledMotion);
els.useCharacter.addEventListener("click", () => useBundledCharacter().catch(handleError));
els.startRecord.addEventListener("click", () => startRecording().catch(handleError));
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
els.playBoth.addEventListener("click", playBoth);
els.pauseBoth.addEventListener("click", pauseBoth);
els.timeline.addEventListener("input", (event) => seekBoth(event.target.value));
els.animationVideo.addEventListener("loadedmetadata", () => {
  syncTimeline();
  renderUi();
});
els.sourceVideo.addEventListener("loadedmetadata", () => {
  syncTimeline();
  renderUi();
});
els.animationVideo.addEventListener("error", () => setStatus(`Animation video failed to load: ${describeVideoError(els.animationVideo)}.`));
els.sourceVideo.addEventListener("error", () => setStatus(`Source video failed to load: ${describeVideoError(els.sourceVideo)}.`));
els.animationVideo.addEventListener("timeupdate", () => {
  els.timeline.value = String(els.animationVideo.currentTime || 0);
  if (hasPlayableVideo(els.sourceVideo) && Math.abs(els.sourceVideo.currentTime - els.animationVideo.currentTime) > 0.2) {
    els.sourceVideo.currentTime = Math.min(els.animationVideo.currentTime, els.sourceVideo.duration || els.animationVideo.currentTime);
  }
});

updateFileNames();
renderUi();
loadAssets().catch(handleError);
