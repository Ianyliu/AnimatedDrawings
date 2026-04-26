export function startGlassPoseAnimation(canvas) {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const points = [
    [0.50, 0.13], [0.42, 0.28], [0.58, 0.28], [0.36, 0.46], [0.64, 0.46],
    [0.31, 0.64], [0.69, 0.64], [0.45, 0.52], [0.55, 0.52], [0.43, 0.76],
    [0.57, 0.76], [0.41, 0.93], [0.59, 0.93],
  ].map(([x, y], index) => ({
    x,
    y,
    px: x + Math.sin(index) * 0.12,
    py: y + Math.cos(index * 1.7) * 0.11,
    delay: index * 0.055,
  }));
  const lines = [[0, 1], [0, 2], [1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];
  let rafId = null;
  let start = performance.now();
  const reducedMotion = prefersReducedMotion();

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    context.setTransform(scale, 0, 0, scale, 0, 0);
    if (reducedMotion) drawFrame(start + 2400);
  }

  function drawFrame(now) {
    const t = (now - start) / 1000;
    const rect = canvas.getBoundingClientRect();
    context.clearRect(0, 0, rect.width, rect.height);
    const size = Math.min(rect.width, rect.height) * 0.86;
    const ox = rect.width * 0.5 - size * 0.5;
    const oy = rect.height * 0.53 - size * 0.5;

    const projected = points.map((point) => {
      const settle = smoothstep(Math.min(1, Math.max(0, (t - point.delay) / 1.4)));
      const drift = Math.sin(t * 1.4 + point.delay * 20) * 0.006;
      const x = point.px + (point.x - point.px) * settle + drift;
      const y = point.py + (point.y - point.py) * settle + Math.cos(t * 1.2 + point.delay * 16) * 0.005;
      return [ox + x * size, oy + y * size, settle];
    });

    const gradient = context.createLinearGradient(0, 0, rect.width, rect.height);
    gradient.addColorStop(0, "rgba(15,118,110,0.64)");
    gradient.addColorStop(0.52, "rgba(255,255,255,0.88)");
    gradient.addColorStop(1, "rgba(39,93,181,0.62)");
    context.strokeStyle = gradient;
    context.lineWidth = 2.8;
    context.lineCap = "round";
    context.shadowColor = "rgba(39, 93, 181, 0.34)";
    context.shadowBlur = 24;
    for (const [a, b] of lines) {
      const alpha = Math.min(projected[a][2], projected[b][2]);
      context.globalAlpha = 0.28 + alpha * 0.72;
      context.beginPath();
      context.moveTo(projected[a][0], projected[a][1]);
      context.lineTo(projected[b][0], projected[b][1]);
      context.stroke();
    }

    for (const [x, y, alpha] of projected) {
      context.globalAlpha = 0.45 + alpha * 0.55;
      context.fillStyle = "rgba(255,255,255,0.94)";
      context.beginPath();
      context.arc(x, y, 6.2, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "rgba(15, 118, 110, 0.62)";
      context.lineWidth = 1.4;
      context.stroke();
    }
    context.globalAlpha = 1;
  }

  function draw(now) {
    drawFrame(now);
    rafId = requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  if (!reducedMotion) {
    rafId = requestAnimationFrame(draw);
  }
  return () => {
    if (rafId) cancelAnimationFrame(rafId);
    window.removeEventListener("resize", resize);
  };
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}
