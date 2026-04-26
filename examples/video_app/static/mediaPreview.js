import { state } from "./workflowState.js";

export function hasPlayableVideo(video) {
  return Boolean(video.currentSrc || video.getAttribute("src"));
}

export function cacheBust(url) {
  if (!url) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}client_ts=${Date.now()}`;
}

export function clearVideo(video, els) {
  video.pause();
  video.removeAttribute("src");
  video.load();
  if (els) syncTimeline(els);
}

export function setVideoSource(video, url, els) {
  clearVideo(video, els);
  if (!url) return;
  video.src = cacheBust(url);
  video.load();
}

export function setImageSource(image, url) {
  if (!url) {
    image.removeAttribute("src");
    return;
  }
  image.src = cacheBust(url);
}

export async function startRecording(els, beginPhaseAndRender, finishRecording) {
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
    finishRecording();
  };

  state.mediaRecorder.start();
  beginPhaseAndRender("recording", `Recording up to ${window.APP_MAX_SECONDS}s...`);
  setTimeout(() => {
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
    }
  }, window.APP_MAX_SECONDS * 1000);
}

export function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
  }
}

export function syncTimeline(els) {
  const durations = [els.animationVideo.duration, els.sourceVideo.duration].filter(Number.isFinite);
  const duration = durations.length ? Math.max(...durations) : 0;
  els.timeline.max = String(duration || 0);
}

export function playBoth(els, setStatus) {
  if (hasPlayableVideo(els.sourceVideo)) els.sourceVideo.play().catch((e) => setStatus(e.message));
  if (hasPlayableVideo(els.animationVideo)) els.animationVideo.play().catch((e) => setStatus(e.message));
}

export function pauseBoth(els) {
  els.sourceVideo.pause();
  els.animationVideo.pause();
}

export function seekBoth(els, value) {
  const time = Number(value);
  if (hasPlayableVideo(els.sourceVideo)) {
    els.sourceVideo.currentTime = Math.min(time, els.sourceVideo.duration || time);
  }
  if (hasPlayableVideo(els.animationVideo)) {
    els.animationVideo.currentTime = Math.min(time, els.animationVideo.duration || time);
  }
}

export function describeVideoError(video) {
  if (!video.error) return "unknown error";
  const messages = {
    1: "loading was aborted",
    2: "network error",
    3: "decode error",
    4: "source not supported",
  };
  return messages[video.error.code] || `error code ${video.error.code}`;
}
