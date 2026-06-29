import {
  DEFAULT_CONFIG,
  GEOMETRIES,
  cloneState,
  createState,
  diagnostics,
  exactStep,
  exportHistoryCsv,
  heatColor,
  makeConfig,
  sourceColor,
  stepMode,
} from "./simulator.js";

const elements = {
  geometryList: document.querySelector("#geometryList"),
  modeButtons: [...document.querySelectorAll("[data-mode]")],
  viewButtons: [...document.querySelectorAll("[data-view]")],
  play: document.querySelector("#play"),
  step: document.querySelector("#step"),
  burst: document.querySelector("#burst"),
  reset: document.querySelector("#reset"),
  exportCsv: document.querySelector("#exportCsv"),
  snapshot: document.querySelector("#snapshot"),
  heatCanvas: document.querySelector("#heatCanvas"),
  sourceCanvas: document.querySelector("#sourceCanvas"),
  plotCanvas: document.querySelector("#plotCanvas"),
  grid: document.querySelector("#grid"),
  boundary: document.querySelector("#boundary"),
  stride: document.querySelector("#stride"),
  diffusivity: document.querySelector("#diffusivity"),
  cooling: document.querySelector("#cooling"),
  source: document.querySelector("#source"),
  frameSteps: document.querySelector("#frameSteps"),
  correction: document.querySelector("#correction"),
  compareExact: document.querySelector("#compareExact"),
  readouts: [...document.querySelectorAll("[data-readout]")],
  modeLabel: document.querySelector("#modeLabel"),
  geometryLabel: document.querySelector("#geometryLabel"),
  methodNote: document.querySelector("#methodNote"),
  tags: document.querySelector("#tags"),
};

const app = {
  config: makeConfig(DEFAULT_CONFIG),
  mode: "ai",
  view: "temperature",
  running: false,
  state: null,
  reference: null,
  history: [],
  lastFrameAt: performance.now(),
  recentStepsPerSecond: 0,
  lastDiagnostics: null,
};

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(digits);
}

function setReadout(name, value) {
  for (const node of elements.readouts) {
    if (node.dataset.readout === name) node.textContent = value;
  }
}

function readConfig() {
  return makeConfig({
    n: Number(elements.grid.value),
    boundary: elements.boundary.value,
    stride: Number(elements.stride.value),
    diffusivity: Number(elements.diffusivity.value),
    baseCooling: Number(elements.cooling.value),
    sourceScale: Number(elements.source.value),
    coolingScale: 1,
    hybridCorrectionInterval: Number(elements.correction.value),
    geometry: app.config.geometry,
  });
}

function syncControlLabels() {
  document.querySelector("#strideValue").textContent = `${elements.stride.value} solver steps`;
  document.querySelector("#diffusivityValue").textContent = Number(elements.diffusivity.value).toFixed(4);
  document.querySelector("#coolingValue").textContent = Number(elements.cooling.value).toFixed(4);
  document.querySelector("#sourceValue").textContent = Number(elements.source.value).toFixed(2);
  document.querySelector("#frameStepsValue").textContent = `${elements.frameSteps.value} macro steps`;
  document.querySelector("#correctionValue").textContent = `every ${elements.correction.value}`;
}

function buildGeometryList() {
  elements.geometryList.textContent = "";
  for (const geometry of GEOMETRIES) {
    const button = document.createElement("button");
    button.className = "geometry-option";
    button.type = "button";
    button.dataset.geometry = geometry.id;
    button.innerHTML = `<strong>${geometry.name}</strong><span>${geometry.short}</span>`;
    button.addEventListener("click", () => {
      app.config = makeConfig({ ...readConfig(), geometry: geometry.id });
      resetSimulation();
    });
    elements.geometryList.append(button);
  }
}

function updateActiveButtons() {
  for (const button of elements.modeButtons) button.classList.toggle("is-active", button.dataset.mode === app.mode);
  for (const button of elements.viewButtons) button.classList.toggle("is-active", button.dataset.view === app.view);
  for (const button of elements.geometryList.querySelectorAll("[data-geometry]")) {
    button.classList.toggle("is-active", button.dataset.geometry === app.config.geometry);
  }
}

function resetSimulation() {
  app.config = readConfig();
  app.state = createState(app.config);
  app.reference = cloneState(app.state);
  app.history = [];
  app.recentStepsPerSecond = 0;
  app.lastDiagnostics = diagnostics(app.state, app.config, app.reference);
  updateActiveButtons();
  updateText();
  render();
}

function updateText() {
  const geometry = GEOMETRIES.find((item) => item.id === app.config.geometry);
  elements.modeLabel.textContent = app.mode === "exact" ? "Exact solver" : app.mode === "ai" ? "AI macro surrogate" : "Hybrid correction";
  elements.geometryLabel.textContent = geometry ? geometry.name : app.config.geometry;
  elements.tags.textContent = app.state.labels.join(" / ");
  const stride = app.config.stride;
  if (app.mode === "exact") {
    elements.methodNote.textContent = `Exact mode advances ${stride} explicit heat-equation steps per visual update. Use this as the reference when checking surrogate drift.`;
  } else if (app.mode === "ai") {
    elements.methodNote.textContent = `AI mode uses a neural-operator-style macro predictor: diffuse low-frequency temperature/source features, inject power, and advance ${stride} solver-equivalent steps per inference.`;
  } else {
    elements.methodNote.textContent = `Hybrid mode uses the AI macro predictor between exact corrections. Every ${app.config.hybridCorrectionInterval} macro steps, the trusted heat solver replaces the surrogate update.`;
  }
}

function advanceOne() {
  if (app.mode === "exact") {
    stepMode("exact", app.state, app.config);
    app.reference.temperature.set(app.state.temperature);
    app.reference.time = app.state.time;
    app.reference.solverSteps = app.state.solverSteps;
  } else {
    stepMode(app.mode, app.state, app.config);
    if (elements.compareExact.checked) exactStep(app.reference, app.config, app.config.stride);
  }
}

function advanceFrame() {
  const count = Number(elements.frameSteps.value);
  const stepsBefore = app.state.solverSteps;
  const start = performance.now();
  for (let i = 0; i < count; i += 1) advanceOne();
  const elapsed = Math.max(0.001, performance.now() - start);
  app.recentStepsPerSecond = (app.state.solverSteps - stepsBefore) / (elapsed / 1000);
  recordHistory();
}

function recordHistory() {
  const reference = elements.compareExact.checked || app.mode === "exact" ? app.reference : null;
  const diag = diagnostics(app.state, app.config, reference);
  const row = {
    ...diag,
    mode: app.mode,
    stepsPerSecond: app.recentStepsPerSecond,
  };
  app.lastDiagnostics = row;
  app.history.push(row);
  if (app.history.length > 260) app.history.shift();
}

function tick(now) {
  if (!app.running) return;
  if (now - app.lastFrameAt > 14) {
    app.lastFrameAt = now;
    advanceFrame();
    render();
  }
  requestAnimationFrame(tick);
}

function fieldForView() {
  if (app.view === "source") return { field: app.state.source, min: 0, max: maxOf(app.state.source), kind: "source" };
  if (app.view === "cooling") return { field: app.state.cooling, min: 0, max: maxOf(app.state.cooling), kind: "source" };
  if (app.view === "error" && elements.compareExact.checked) {
    const error = new Float32Array(app.state.temperature.length);
    let max = 0;
    for (let i = 0; i < error.length; i += 1) {
      error[i] = Math.abs(app.state.temperature[i] - app.reference.temperature[i]);
      max = Math.max(max, error[i]);
    }
    return { field: error, min: 0, max, kind: "temperature" };
  }
  return { field: app.state.temperature, min: Math.min(0, minOf(app.state.temperature)), max: Math.max(1e-6, maxOf(app.state.temperature)), kind: "temperature" };
}

function minOf(field) {
  let value = Infinity;
  for (const item of field) value = Math.min(value, item);
  return value;
}

function maxOf(field) {
  let value = -Infinity;
  for (const item of field) value = Math.max(value, item);
  return value;
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * scale));
  const height = Math.max(1, Math.floor(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawField(canvas, field, min, max, kind) {
  const { ctx, width, height } = prepareCanvas(canvas);
  const n = app.config.n;
  const image = ctx.createImageData(n, n);
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const sourceIndex = y * n + x;
      const target = sourceIndex * 4;
      const color = kind === "source" ? sourceColor(field[sourceIndex], max) : heatColor(field[sourceIndex], min, max);
      image.data[target] = color[0];
      image.data[target + 1] = color[1];
      image.data[target + 2] = color[2];
      image.data[target + 3] = 255;
    }
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = n;
  offscreen.height = n;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = false;
  const size = Math.min(width, height);
  const ox = (width - size) * 0.5;
  const oy = (height - size) * 0.5;
  ctx.drawImage(offscreen, ox, oy, size, size);
  ctx.strokeStyle = "rgba(17, 24, 39, 0.32)";
  ctx.lineWidth = 1;
  ctx.strokeRect(ox + 0.5, oy + 0.5, size - 1, size - 1);
}

function drawPlot() {
  const { ctx, width, height } = prepareCanvas(elements.plotCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  if (app.history.length < 2) return;

  const series = [
    { key: "max", label: "max rise", color: "#c2410c" },
    { key: "mean", label: "mean rise", color: "#1d4ed8" },
    { key: "relativeError", label: "relative error", color: "#0f766e" },
  ];
  const pad = { left: 44, right: 18, top: 20, bottom: 32 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.strokeStyle = "#d8dee7";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  const maxY = Math.max(
    1e-6,
    ...app.history.flatMap((row) => series.map((s) => Math.abs(row[s.key] || 0)))
  );
  const xFor = (i) => pad.left + (plotW * i) / Math.max(1, app.history.length - 1);
  const yFor = (value) => pad.top + plotH - (plotH * value) / maxY;

  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#64748b";
  ctx.fillText("0", 10, pad.top + plotH + 4);
  ctx.fillText(formatNumber(maxY, 2), 10, pad.top + 4);

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    app.history.forEach((row, index) => {
      const x = xFor(index);
      const y = yFor(Math.max(0, row[item.key] || 0));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  let legendX = pad.left;
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, height - 18, 10, 10);
    ctx.fillStyle = "#334155";
    ctx.fillText(item.label, legendX + 14, height - 9);
    legendX += 104;
  }
}

function updateMetrics() {
  const diag = app.lastDiagnostics || diagnostics(app.state, app.config, app.reference);
  setReadout("max", `${formatNumber(diag.max, 3)} K`);
  setReadout("mean", `${formatNumber(diag.mean, 3)} K`);
  setReadout("absolute", `${formatNumber(diag.absoluteMax, 2)} K`);
  setReadout("gradient", formatNumber(diag.gradientRms, 3));
  setReadout("error", elements.compareExact.checked ? formatNumber(diag.relativeError, 4) : "off");
  setReadout("steps", `${diag.solverSteps}`);
  setReadout("time", formatNumber(diag.time, 3));
  setReadout("speed", `${formatNumber(app.recentStepsPerSecond, 0)} steps/s`);
  setReadout("netPower", formatNumber(diag.netPower, 4));
  setReadout("lastMode", diag.lastMode);
}

function render() {
  const view = fieldForView();
  drawField(elements.heatCanvas, view.field, view.min, view.max, view.kind);
  drawField(elements.sourceCanvas, app.state.source, 0, maxOf(app.state.source), "source");
  drawPlot();
  updateMetrics();
  updateText();
}

function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  for (const button of elements.modeButtons) {
    button.addEventListener("click", () => {
      app.mode = button.dataset.mode;
      updateActiveButtons();
      updateText();
    });
  }
  for (const button of elements.viewButtons) {
    button.addEventListener("click", () => {
      app.view = button.dataset.view;
      updateActiveButtons();
      render();
    });
  }
  elements.play.addEventListener("click", () => {
    app.running = !app.running;
    elements.play.textContent = app.running ? "Pause" : "Run";
    app.lastFrameAt = performance.now();
    if (app.running) requestAnimationFrame(tick);
  });
  elements.step.addEventListener("click", () => {
    advanceFrame();
    render();
  });
  elements.burst.addEventListener("click", () => {
    const previous = Number(elements.frameSteps.value);
    elements.frameSteps.value = 64;
    advanceFrame();
    elements.frameSteps.value = previous;
    syncControlLabels();
    render();
  });
  elements.reset.addEventListener("click", resetSimulation);
  elements.exportCsv.addEventListener("click", () => download("heat-lab-history.csv", exportHistoryCsv(app.history), "text/csv"));
  elements.snapshot.addEventListener("click", () => {
    const url = elements.heatCanvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    link.download = "heat-lab-field.png";
    link.click();
  });

  const resetControls = [elements.grid, elements.boundary];
  for (const control of resetControls) control.addEventListener("change", resetSimulation);

  const liveControls = [elements.stride, elements.diffusivity, elements.cooling, elements.source, elements.frameSteps, elements.correction];
  for (const control of liveControls) {
    control.addEventListener("input", () => {
      syncControlLabels();
      app.config = readConfig();
      if (control === elements.source) resetSimulation();
      else render();
    });
  }
  elements.compareExact.addEventListener("change", () => {
    app.reference = cloneState(app.state);
    if (elements.compareExact.checked && app.mode !== "exact") {
      app.reference = createState(app.config);
      exactStep(app.reference, app.config, app.state.solverSteps);
    }
    render();
  });
  window.addEventListener("resize", render);
}

buildGeometryList();
syncControlLabels();
bindEvents();
resetSimulation();
