const FIGURE_POINTS = [
  [0.50, 0.08],
  [0.42, 0.24],
  [0.58, 0.24],
  [0.35, 0.43],
  [0.65, 0.43],
  [0.29, 0.62],
  [0.71, 0.62],
  [0.44, 0.50],
  [0.56, 0.50],
  [0.41, 0.73],
  [0.59, 0.73],
  [0.35, 0.92],
  [0.65, 0.92],
];

const FIGURE_LINES = [
  [0, 1],
  [0, 2],
  [1, 2],
  [1, 3],
  [3, 5],
  [2, 4],
  [4, 6],
  [1, 7],
  [2, 8],
  [7, 8],
  [7, 9],
  [9, 11],
  [8, 10],
  [10, 12],
];

const COLORS = ["#42d7c2", "#79a8ff", "#ffffff", "#ffd66e", "#ff7a90"];

export function startOpeningExperience({ screen, canvas, skipButton, maxDurationMs = 4600 } = {}) {
  if (!screen || !canvas) return Promise.resolve();

  if (prefersReducedMotion()) {
    screen.classList.add("openingDone");
    screen.hidden = true;
    return Promise.resolve();
  }

  const context = canvas.getContext("2d");
  if (!context) {
    screen.hidden = true;
    return Promise.resolve();
  }

  document.documentElement.classList.add("openingActive");
  screen.hidden = false;

  const streaks = makeStreaks(64);
  const start = performance.now();
  const pointer = { x: 0.5, y: 0.5, active: false };
  let rafId = 0;
  let finishTimer = 0;
  let done = false;

  return new Promise((resolve) => {
    const finish = () => {
      if (done) return;
      done = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (finishTimer) window.clearTimeout(finishTimer);
      cleanup();
      screen.classList.add("openingDone");
      document.documentElement.classList.remove("openingActive");
      window.setTimeout(() => {
        screen.hidden = true;
        resolve();
      }, 540);
    };

    const onKeydown = (event) => {
      if (event.key === "Escape" || event.key === "Enter" || event.key === " ") finish();
    };
    const onPointerMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = clamp((event.clientX - rect.left) / Math.max(1, rect.width));
      pointer.y = clamp((event.clientY - rect.top) / Math.max(1, rect.height));
      pointer.active = true;
    };
    const onPointerLeave = () => {
      pointer.active = false;
    };
    const onResize = () => resizeCanvas(canvas, context);

    function cleanup() {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKeydown);
      screen.removeEventListener("pointermove", onPointerMove);
      screen.removeEventListener("pointerleave", onPointerLeave);
      skipButton?.removeEventListener("click", finish);
    }

    function tick(now) {
      const elapsed = now - start;
      drawOpeningFrame(context, canvas, streaks, elapsed, maxDurationMs, pointer);
      if (elapsed >= maxDurationMs) {
        finish();
        return;
      }
      rafId = requestAnimationFrame(tick);
    }

    resizeCanvas(canvas, context);
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKeydown);
    screen.addEventListener("pointermove", onPointerMove);
    screen.addEventListener("pointerleave", onPointerLeave);
    skipButton?.addEventListener("click", finish);
    finishTimer = window.setTimeout(finish, maxDurationMs + 280);
    rafId = requestAnimationFrame(tick);
  });
}

function drawOpeningFrame(context, canvas, streaks, elapsed, maxDurationMs, pointer) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const progress = clamp(elapsed / maxDurationMs);
  const gather = smoothstep(clamp(elapsed / 1350));
  const burst = smoothstep(clamp((elapsed - 1180) / 850));
  const reveal = 0.42 + smoothstep(clamp((elapsed - 240) / 920)) * 0.58;
  const curtain = smoothstep(clamp((elapsed - 3600) / 800));
  const stageX = width > 760 ? 0.69 : 0.50;
  const centerX = width * (stageX + (pointer.active ? (pointer.x - 0.5) * 0.035 : 0));
  const centerY = height * (0.51 + (pointer.active ? (pointer.y - 0.5) * 0.025 : 0));

  context.clearRect(0, 0, width, height);
  drawBackdropSweep(context, width, height, progress, curtain);
  drawStreaks(context, streaks, width, height, centerX, centerY, gather, burst, elapsed);
  drawFlash(context, width, height, centerX, centerY, elapsed);
  drawJumpingFigure(context, width, height, centerX, centerY, reveal, elapsed, pointer);
  drawCurtainLift(context, width, height, curtain);
}

function drawBackdropSweep(context, width, height, progress, curtain) {
  context.save();
  context.globalAlpha = 0.68 * (1 - curtain);
  const sweepX = width * (-0.25 + progress * 1.5);
  const gradient = context.createLinearGradient(sweepX - width * 0.3, 0, sweepX + width * 0.3, height);
  gradient.addColorStop(0, "rgba(66, 215, 194, 0)");
  gradient.addColorStop(0.34, "rgba(66, 215, 194, 0.24)");
  gradient.addColorStop(0.52, "rgba(255, 255, 255, 0.28)");
  gradient.addColorStop(0.72, "rgba(255, 214, 110, 0.20)");
  gradient.addColorStop(1, "rgba(121, 168, 255, 0)");
  context.fillStyle = gradient;
  context.translate(sweepX, height * 0.5);
  context.rotate(-0.34);
  context.fillRect(-width * 0.34, -height, width * 0.68, height * 2);
  context.restore();
}

function drawStreaks(context, streaks, width, height, centerX, centerY, gather, burst, elapsed) {
  context.save();
  context.lineCap = "round";
  context.globalCompositeOperation = "screen";
  for (const streak of streaks) {
    const localGather = smoothstep(clamp((gather - streak.delay) / 0.82));
    const localBurst = smoothstep(clamp((burst - streak.delay * 0.55) / 0.85));
    const targetX = centerX + streak.targetX * width * 0.18;
    const targetY = centerY + streak.targetY * height * 0.18;
    const endX = centerX + streak.endX * width * 0.46;
    const endY = centerY + streak.endY * height * 0.42;
    const headX = lerp(lerp(streak.startX * width, targetX, localGather), endX, localBurst);
    const headY = lerp(lerp(streak.startY * height, targetY, localGather), endY, localBurst);
    const trail = 18 + streak.length * 34;
    const angle = Math.atan2(headY - targetY, headX - targetX) + Math.sin(elapsed * 0.002 + streak.seed) * 0.18;
    const tailX = headX - Math.cos(angle) * trail;
    const tailY = headY - Math.sin(angle) * trail;
    context.strokeStyle = streak.color;
    context.lineWidth = 1.1 + streak.weight * 2.6;
    context.globalAlpha = (0.18 + localGather * 0.72) * (1 - localBurst * 0.36);
    context.beginPath();
    context.moveTo(tailX, tailY);
    context.lineTo(headX, headY);
    context.stroke();
  }
  context.restore();
}

function drawFlash(context, width, height, centerX, centerY, elapsed) {
  const flash = smoothstep(clamp((elapsed - 360) / 220)) * (1 - smoothstep(clamp((elapsed - 960) / 620)));
  const ring = smoothstep(clamp((elapsed - 480) / 760)) * (1 - smoothstep(clamp((elapsed - 2100) / 720)));
  if (flash <= 0 && ring <= 0) return;

  context.save();
  context.globalCompositeOperation = "screen";
  const burstGradient = context.createRadialGradient(centerX, centerY, 0, centerX, centerY, Math.min(width, height) * 0.42);
  burstGradient.addColorStop(0, `rgba(255, 255, 255, ${0.54 * flash})`);
  burstGradient.addColorStop(0.22, `rgba(66, 215, 194, ${0.24 * flash})`);
  burstGradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  context.fillStyle = burstGradient;
  context.fillRect(0, 0, width, height);

  context.globalAlpha = ring * 0.78;
  context.strokeStyle = "rgba(255, 255, 255, 0.9)";
  context.lineWidth = 2.4;
  context.beginPath();
  context.arc(centerX, centerY, Math.min(width, height) * (0.06 + ring * 0.30), 0, Math.PI * 2);
  context.stroke();
  context.restore();
}

function drawJumpingFigure(context, width, height, centerX, centerY, reveal, elapsed, pointer) {
  if (reveal <= 0) return;

  const jumpPhase = clamp((elapsed - 380) / 2300);
  const jump = Math.sin(jumpPhase * Math.PI) * Math.min(width, height) * 0.08;
  const landPulse = 1 + (1 - smoothstep(clamp((elapsed - 3300) / 500))) * 0.045;
  const size = Math.min(width, height) * 0.38 * landPulse;
  const pointerTilt = pointer.active ? (pointer.x - 0.5) * 0.05 : 0;
  const projected = FIGURE_POINTS.map(([x, y], index) => {
    const wave = Math.sin(elapsed * 0.004 + index * 0.72) * 0.014;
    const px = centerX + (x - 0.5 + wave + pointerTilt) * size;
    const py = centerY - jump + (y - 0.5) * size;
    return [px, py];
  });

  context.save();
  context.globalAlpha = reveal;
  context.globalCompositeOperation = "source-over";
  context.lineCap = "round";
  context.lineJoin = "round";
  const gradient = context.createLinearGradient(centerX - size * 0.55, centerY - size * 0.5, centerX + size * 0.55, centerY + size * 0.5);
  gradient.addColorStop(0, "rgba(66, 215, 194, 0.96)");
  gradient.addColorStop(0.42, "rgba(18, 28, 38, 0.86)");
  gradient.addColorStop(1, "rgba(121, 168, 255, 0.94)");
  context.strokeStyle = gradient;
  context.lineWidth = Math.max(2.4, size * 0.014);
  context.shadowColor = "rgba(121, 168, 255, 0.46)";
  context.shadowBlur = 28;
  for (const [a, b] of FIGURE_LINES) {
    context.beginPath();
    context.moveTo(projected[a][0], projected[a][1]);
    context.lineTo(projected[b][0], projected[b][1]);
    context.stroke();
  }

  for (const [x, y] of projected) {
    context.fillStyle = "rgba(255, 255, 255, 0.94)";
    context.beginPath();
    context.arc(x, y, Math.max(4, size * 0.021), 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "rgba(18, 28, 38, 0.52)";
    context.lineWidth = 1.4;
    context.stroke();
  }

  const floorY = centerY + size * 0.48;
  const floorAlpha = smoothstep(clamp((elapsed - 3000) / 650));
  context.globalAlpha = reveal * floorAlpha * 0.58;
  context.shadowBlur = 18;
  context.strokeStyle = "rgba(255, 255, 255, 0.76)";
  context.lineWidth = 2;
  context.beginPath();
  context.ellipse(centerX, floorY, size * 0.36, size * 0.035, 0, 0, Math.PI * 2);
  context.stroke();
  context.restore();
}

function drawCurtainLift(context, width, height, curtain) {
  if (curtain <= 0) return;
  context.save();
  context.globalAlpha = curtain * 0.56;
  const gradient = context.createLinearGradient(0, height * (0.45 - curtain * 0.2), 0, height);
  gradient.addColorStop(0, "rgba(255, 255, 255, 0)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0.82)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  context.restore();
}

function makeStreaks(count) {
  return Array.from({ length: count }, (_, index) => {
    const side = index % 4;
    const edge = seeded(index + 1);
    const angle = seeded(index + 11) * Math.PI * 2;
    return {
      startX: side === 0 ? -0.08 : side === 1 ? 1.08 : edge,
      startY: side === 2 ? -0.08 : side === 3 ? 1.08 : edge,
      targetX: Math.cos(angle) * seeded(index + 21),
      targetY: Math.sin(angle) * seeded(index + 31),
      endX: Math.cos(angle) * (0.35 + seeded(index + 41) * 0.9),
      endY: Math.sin(angle) * (0.30 + seeded(index + 51) * 0.8),
      length: seeded(index + 61),
      weight: seeded(index + 71),
      delay: seeded(index + 81) * 0.28,
      color: COLORS[index % COLORS.length],
      seed: index * 1.97,
    };
  });
}

function resizeCanvas(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * scale));
  canvas.height = Math.max(1, Math.round(rect.height * scale));
  context.setTransform(scale, 0, 0, scale, 0, 0);
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function seeded(value) {
  const x = Math.sin(value * 9283.17) * 43758.5453;
  return x - Math.floor(x);
}

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

function lerp(a, b, amount) {
  return a + (b - a) * amount;
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}
