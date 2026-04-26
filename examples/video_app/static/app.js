const state = {
  sessionId: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
  recordedBlob: null,
  mediaRecorder: null,
  stream: null,
  motion: null,
  drawing: null,
};

const els = {
  motionSelect: document.getElementById("motionSelect"),
  useMotion: document.getElementById("useMotion"),
  bvhUpload: document.getElementById("bvhUpload"),
  uploadBvh: document.getElementById("uploadBvh"),
  recordPreview: document.getElementById("recordPreview"),
  startRecord: document.getElementById("startRecord"),
  stopRecord: document.getElementById("stopRecord"),
  videoUpload: document.getElementById("videoUpload"),
  processVideo: document.getElementById("processVideo"),
  characterSelect: document.getElementById("characterSelect"),
  useCharacter: document.getElementById("useCharacter"),
  drawingUpload: document.getElementById("drawingUpload"),
  uploadDrawing: document.getElementById("uploadDrawing"),
  jointOverlay: document.getElementById("jointOverlay"),
  renderAnimation: document.getElementById("renderAnimation"),
  status: document.getElementById("status"),
  sourceVideo: document.getElementById("sourceVideo"),
  animationVideo: document.getElementById("animationVideo"),
  playBoth: document.getElementById("playBoth"),
  pauseBoth: document.getElementById("pauseBoth"),
  timeline: document.getElementById("timeline"),
};

function setStatus(message) {
  els.status.textContent = message || "";
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
  image.src = cacheBust(url);
}

async function postForm(url, form) {
  form.append("session_id", state.sessionId);
  const response = await fetch(url, { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, session_id: state.sessionId }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function loadAssets() {
  const response = await fetch("/api/assets");
  const assets = await response.json();

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
    option.textContent = character.name;
    els.characterSelect.appendChild(option);
  }

  useBundledMotion();
}

function useBundledMotion() {
  const selected = els.motionSelect.selectedOptions[0];
  if (!selected) return;
  state.motion = {
    motion_cfg: selected.value,
    retarget_cfg: selected.dataset.retarget,
    overlay_url: null,
  };
  clearVideo(els.sourceVideo);
  setStatus("Motion selected.");
}

async function useBundledCharacter() {
  const selected = els.characterSelect.selectedOptions[0];
  if (!selected) return;
  setStatus("Loading character joints...");
  const form = new FormData();
  form.append("character_cfg", selected.value);
  const data = await postForm("/api/drawing", form);
  state.drawing = data;
  setImageSource(els.jointOverlay, data.joint_overlay_url);
  setStatus("Character ready.");
}

async function startRecording() {
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
    els.recordPreview.srcObject = null;
    els.recordPreview.src = URL.createObjectURL(state.recordedBlob);
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    els.startRecord.disabled = false;
    els.stopRecord.disabled = true;
    setStatus("Recording ready.");
  };

  state.mediaRecorder.start();
  els.startRecord.disabled = true;
  els.stopRecord.disabled = false;
  setStatus(`Recording up to ${window.APP_MAX_SECONDS}s...`);
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
  setStatus("Estimating pose...");
  const data = await postForm("/api/motion/video", form);
  state.motion = data;
  setVideoSource(els.sourceVideo, data.overlay_url);
  setStatus("Video motion ready.");
}

async function uploadBvh() {
  const file = els.bvhUpload.files[0];
  if (!file) {
    setStatus("Select a BVH file first.");
    return;
  }
  const form = new FormData();
  form.append("bvh", file);
  setStatus("Preparing BVH...");
  const data = await postForm("/api/motion/bvh", form);
  state.motion = data;
  clearVideo(els.sourceVideo);
  setStatus("BVH motion ready.");
}

async function uploadDrawing() {
  const file = els.drawingUpload.files[0];
  if (!file) {
    setStatus("Select a drawing first.");
    return;
  }
  const form = new FormData();
  form.append("drawing", file);
  setStatus("Estimating drawing joints...");
  const data = await postForm("/api/drawing", form);
  state.drawing = data;
  setImageSource(els.jointOverlay, data.joint_overlay_url);
  setStatus("Drawing ready.");
}

async function renderAnimation() {
  if (!state.motion || !state.drawing) {
    setStatus("Choose a motion and drawing first.");
    return;
  }
  setStatus("Rendering animation...");
  const data = await postJson("/api/render", {
    character_cfg: state.drawing.character_cfg,
    motion_cfg: state.motion.motion_cfg,
    retarget_cfg: state.motion.retarget_cfg,
  });
  setVideoSource(els.animationVideo, data.animation_url);
  setStatus("Animation ready.");
}

function syncTimeline() {
  const durations = [els.animationVideo.duration, els.sourceVideo.duration].filter(Number.isFinite);
  const duration = durations.length ? Math.max(...durations) : 0;
  els.timeline.max = String(duration || 0);
}

function playBoth() {
  if (els.sourceVideo.currentSrc) els.sourceVideo.play().catch((e) => setStatus(e.message));
  if (els.animationVideo.currentSrc) els.animationVideo.play().catch((e) => setStatus(e.message));
}

function pauseBoth() {
  els.sourceVideo.pause();
  els.animationVideo.pause();
}

function seekBoth(value) {
  const time = Number(value);
  if (els.sourceVideo.currentSrc) els.sourceVideo.currentTime = Math.min(time, els.sourceVideo.duration || time);
  if (els.animationVideo.currentSrc) els.animationVideo.currentTime = Math.min(time, els.animationVideo.duration || time);
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

els.useMotion.addEventListener("click", useBundledMotion);
els.useCharacter.addEventListener("click", () => useBundledCharacter().catch((e) => setStatus(e.message)));
els.startRecord.addEventListener("click", () => startRecording().catch((e) => setStatus(e.message)));
els.stopRecord.addEventListener("click", stopRecording);
els.processVideo.addEventListener("click", () => processVideo().catch((e) => setStatus(e.message)));
els.uploadBvh.addEventListener("click", () => uploadBvh().catch((e) => setStatus(e.message)));
els.uploadDrawing.addEventListener("click", () => uploadDrawing().catch((e) => setStatus(e.message)));
els.renderAnimation.addEventListener("click", () => renderAnimation().catch((e) => setStatus(e.message)));
els.playBoth.addEventListener("click", playBoth);
els.pauseBoth.addEventListener("click", pauseBoth);
els.timeline.addEventListener("input", (event) => seekBoth(event.target.value));
els.animationVideo.addEventListener("loadedmetadata", syncTimeline);
els.sourceVideo.addEventListener("loadedmetadata", syncTimeline);
els.animationVideo.addEventListener("error", () => setStatus(`Animation video failed to load: ${describeVideoError(els.animationVideo)}.`));
els.sourceVideo.addEventListener("error", () => setStatus(`Source video failed to load: ${describeVideoError(els.sourceVideo)}.`));
els.animationVideo.addEventListener("timeupdate", () => {
  els.timeline.value = String(els.animationVideo.currentTime || 0);
  if (els.sourceVideo.currentSrc && Math.abs(els.sourceVideo.currentTime - els.animationVideo.currentTime) > 0.2) {
    els.sourceVideo.currentTime = Math.min(els.animationVideo.currentTime, els.sourceVideo.duration || els.animationVideo.currentTime);
  }
});

loadAssets().catch((e) => setStatus(e.message));
