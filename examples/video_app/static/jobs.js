import { requestJson } from "./api.js";
import { clearJobProgress, renderUi, setStatus, updateJobProgress } from "./renderUi.js";
import { markFailedStep, setPhase, state } from "./workflowState.js";

export async function trackJob(job, phase, onComplete) {
  if (!job) throw new Error("Server did not return a job.");
  clearPollTimer();
  state.activeJob = job;
  setPhase(phase, job.message || "Queued.");
  renderUi();

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
        renderUi();
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
          renderUi();
          reject(e);
        }
      }, 700);
    };

    consume(job);
  });
}

export function clearPollTimer() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}
