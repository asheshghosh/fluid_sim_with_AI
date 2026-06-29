const TWO_PI = Math.PI * 2.0;

export const GEOMETRIES = [
  {
    id: "chiplets",
    name: "Chiplet floorplan",
    short: "Four compute islands with cooled package edges.",
  },
  {
    id: "hotspots",
    name: "Gaussian hotspots",
    short: "A cluster of localized transient power peaks.",
  },
  {
    id: "microchannels",
    name: "Microchannel cooler",
    short: "Heated blocks separated by cold-flow channels.",
  },
  {
    id: "vias",
    name: "Thermal via array",
    short: "Broad die heating with periodic vertical heat sinks.",
  },
  {
    id: "ring",
    name: "Ring heater",
    short: "Annular heater around a cooled sensor island.",
  },
  {
    id: "lshape",
    name: "L-shaped accelerator",
    short: "Asymmetric power island, useful for anisotropic spreading.",
  },
];

export const DEFAULT_CONFIG = Object.freeze({
  n: 96,
  dt: 0.018,
  diffusivity: 0.006,
  baseCooling: 0.003,
  sourceScale: 1.0,
  coolingScale: 1.0,
  boundary: "insulated",
  geometry: "chiplets",
  stride: 8,
  hybridCorrectionInterval: 5,
  ambient: 300.0,
});

function indexOf(x, y, n) {
  return y * n + x;
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function smoothstep(edge0, edge1, value) {
  const t = clamp((value - edge0) / Math.max(1e-12, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function addRect(field, n, cx, cy, width, height, value, edge = 0.012) {
  for (let y = 0; y < n; y += 1) {
    const yy = (y + 0.5) / n * 2 - 1;
    const dy = Math.abs(yy - cy);
    const wy = 1 - smoothstep(height * 0.5 - edge, height * 0.5 + edge, dy);
    if (wy <= 0) continue;
    for (let x = 0; x < n; x += 1) {
      const xx = (x + 0.5) / n * 2 - 1;
      const dx = Math.abs(xx - cx);
      const wx = 1 - smoothstep(width * 0.5 - edge, width * 0.5 + edge, dx);
      if (wx > 0) field[indexOf(x, y, n)] += value * wx * wy;
    }
  }
}

function addGaussian(field, n, cx, cy, sigma, value) {
  const inv = 1 / (2 * sigma * sigma);
  for (let y = 0; y < n; y += 1) {
    const yy = (y + 0.5) / n * 2 - 1;
    for (let x = 0; x < n; x += 1) {
      const xx = (x + 0.5) / n * 2 - 1;
      const dx = xx - cx;
      const dy = yy - cy;
      field[indexOf(x, y, n)] += value * Math.exp(-(dx * dx + dy * dy) * inv);
    }
  }
}

function addRing(field, n, cx, cy, radius, thickness, value) {
  const half = thickness * 0.5;
  for (let y = 0; y < n; y += 1) {
    const yy = (y + 0.5) / n * 2 - 1;
    for (let x = 0; x < n; x += 1) {
      const xx = (x + 0.5) / n * 2 - 1;
      const r = Math.hypot(xx - cx, yy - cy);
      const band = 1 - smoothstep(half * 0.65, half, Math.abs(r - radius));
      if (band > 0) field[indexOf(x, y, n)] += value * band;
    }
  }
}

function addLineCooling(cooling, n, axis, position, width, value) {
  for (let y = 0; y < n; y += 1) {
    const yy = (y + 0.5) / n * 2 - 1;
    for (let x = 0; x < n; x += 1) {
      const xx = (x + 0.5) / n * 2 - 1;
      const d = Math.abs((axis === "x" ? xx : yy) - position);
      const w = 1 - smoothstep(width * 0.45, width * 0.55, d);
      if (w > 0) cooling[indexOf(x, y, n)] += value * w;
    }
  }
}

function normalizePositive(field, targetMax) {
  let maxValue = 0;
  for (const value of field) maxValue = Math.max(maxValue, value);
  if (maxValue <= 0) return field;
  const scale = targetMax / maxValue;
  for (let i = 0; i < field.length; i += 1) field[i] *= scale;
  return field;
}

export function makeConfig(overrides = {}) {
  const config = { ...DEFAULT_CONFIG, ...overrides };
  config.n = Number(config.n);
  config.dt = Number(config.dt);
  config.diffusivity = Number(config.diffusivity);
  config.baseCooling = Number(config.baseCooling);
  config.sourceScale = Number(config.sourceScale);
  config.coolingScale = Number(config.coolingScale);
  config.stride = Math.max(1, Math.round(Number(config.stride)));
  config.hybridCorrectionInterval = Math.max(1, Math.round(Number(config.hybridCorrectionInterval)));
  config.dx = 2 / config.n;
  config.dx2 = config.dx * config.dx;
  return config;
}

export function createGeometry(configLike) {
  const config = makeConfig(configLike);
  const { n } = config;
  const source = new Float32Array(n * n);
  const cooling = new Float32Array(n * n);
  const labels = [];

  if (config.geometry === "chiplets") {
    addRect(source, n, -0.48, -0.34, 0.42, 0.36, 0.85);
    addRect(source, n, 0.34, -0.35, 0.36, 0.34, 1.0);
    addRect(source, n, -0.22, 0.34, 0.34, 0.30, 0.72);
    addRect(source, n, 0.48, 0.30, 0.30, 0.38, 0.58);
    addLineCooling(cooling, n, "x", -0.92, 0.08, 0.75);
    addLineCooling(cooling, n, "x", 0.92, 0.08, 0.75);
    addLineCooling(cooling, n, "y", -0.92, 0.08, 0.75);
    addLineCooling(cooling, n, "y", 0.92, 0.08, 0.75);
    labels.push("rectangular chiplets", "cooled package frame");
  } else if (config.geometry === "hotspots") {
    addGaussian(source, n, -0.50, -0.20, 0.12, 1.0);
    addGaussian(source, n, -0.18, 0.18, 0.10, 0.78);
    addGaussian(source, n, 0.24, -0.08, 0.14, 0.92);
    addGaussian(source, n, 0.55, 0.42, 0.11, 0.64);
    addGaussian(source, n, 0.02, 0.62, 0.18, 0.45);
    labels.push("localized Gaussian power peaks");
  } else if (config.geometry === "microchannels") {
    addRect(source, n, -0.56, 0.0, 0.28, 1.45, 0.72);
    addRect(source, n, 0.0, 0.0, 0.30, 1.45, 1.0);
    addRect(source, n, 0.56, 0.0, 0.28, 1.45, 0.78);
    [-0.82, -0.28, 0.28, 0.82].forEach((x) => addLineCooling(cooling, n, "x", x, 0.10, 1.0));
    labels.push("power lanes", "liquid cooling channels");
  } else if (config.geometry === "vias") {
    addRect(source, n, 0.0, 0.0, 1.42, 1.18, 0.72);
    addGaussian(source, n, -0.35, 0.28, 0.20, 0.55);
    addGaussian(source, n, 0.38, -0.25, 0.18, 0.65);
    for (let yy = -0.6; yy <= 0.61; yy += 0.3) {
      for (let xx = -0.6; xx <= 0.61; xx += 0.3) addGaussian(cooling, n, xx, yy, 0.045, 1.0);
    }
    labels.push("broad die heating", "thermal via sink lattice");
  } else if (config.geometry === "ring") {
    addRing(source, n, 0, 0, 0.54, 0.16, 1.0);
    addGaussian(cooling, n, 0, 0, 0.20, 1.0);
    addLineCooling(cooling, n, "y", -0.92, 0.08, 0.6);
    labels.push("annular heater", "cooled center island");
  } else if (config.geometry === "lshape") {
    addRect(source, n, -0.25, -0.42, 1.0, 0.28, 0.85);
    addRect(source, n, -0.62, 0.05, 0.26, 1.16, 0.95);
    addGaussian(source, n, -0.60, -0.42, 0.10, 0.9);
    addGaussian(source, n, 0.32, -0.40, 0.13, 0.55);
    addLineCooling(cooling, n, "x", 0.78, 0.14, 0.65);
    addLineCooling(cooling, n, "y", 0.80, 0.14, 0.65);
    labels.push("L-shaped accelerator island", "asymmetric heat spreading");
  }

  normalizePositive(source, config.sourceScale);
  normalizePositive(cooling, 0.028 * config.coolingScale);
  return { source, cooling, labels };
}

export function createState(configLike) {
  const config = makeConfig(configLike);
  const geometry = createGeometry(config);
  return {
    n: config.n,
    temperature: new Float32Array(config.n * config.n),
    source: geometry.source,
    cooling: geometry.cooling,
    labels: geometry.labels,
    time: 0,
    solverSteps: 0,
    macroSteps: 0,
    lastMode: "exact",
    _buffer: new Float32Array(config.n * config.n),
    _scratch: new Float32Array(config.n * config.n),
    _scratch2: new Float32Array(config.n * config.n),
  };
}

export function cloneState(state) {
  return {
    n: state.n,
    temperature: new Float32Array(state.temperature),
    source: state.source,
    cooling: state.cooling,
    labels: [...state.labels],
    time: state.time,
    solverSteps: state.solverSteps,
    macroSteps: state.macroSteps,
    lastMode: state.lastMode,
    _buffer: new Float32Array(state.temperature.length),
    _scratch: new Float32Array(state.temperature.length),
    _scratch2: new Float32Array(state.temperature.length),
  };
}

function sample(field, n, x, y, boundary) {
  if (boundary === "periodic") {
    const xx = (x + n) % n;
    const yy = (y + n) % n;
    return field[indexOf(xx, yy, n)];
  }
  if (boundary === "fixed" && (x < 0 || x >= n || y < 0 || y >= n)) return 0;
  const xx = clamp(x, 0, n - 1);
  const yy = clamp(y, 0, n - 1);
  return field[indexOf(xx, yy, n)];
}

function laplacianAt(field, n, x, y, config) {
  const center = field[indexOf(x, y, n)];
  return (
    sample(field, n, x - 1, y, config.boundary) +
    sample(field, n, x + 1, y, config.boundary) +
    sample(field, n, x, y - 1, config.boundary) +
    sample(field, n, x, y + 1, config.boundary) -
    4 * center
  ) / config.dx2;
}

function isFixedEdge(config, x, y) {
  return config.boundary === "fixed" && (x === 0 || y === 0 || x === config.n - 1 || y === config.n - 1);
}

export function exactStep(state, configLike, steps = 1) {
  const config = makeConfig(configLike);
  if (steps <= 0) return state;
  const { n } = config;
  let current = state.temperature;
  let next = state._buffer;

  for (let step = 0; step < steps; step += 1) {
    for (let y = 0; y < n; y += 1) {
      for (let x = 0; x < n; x += 1) {
        const i = indexOf(x, y, n);
        if (isFixedEdge(config, x, y)) {
          next[i] = 0;
          continue;
        }
        const temp = current[i];
        const gamma = config.baseCooling + state.cooling[i];
        const rhs = config.diffusivity * laplacianAt(current, n, x, y, config) - gamma * temp + state.source[i];
        next[i] = temp + config.dt * rhs;
      }
    }
    const swap = current;
    current = next;
    next = swap;
  }

  if (state.temperature !== current) {
    state._buffer = state.temperature;
    state.temperature = current;
  }
  state.time += config.dt * steps;
  state.solverSteps += steps;
  state.macroSteps += 1;
  state.lastMode = "exact";
  return state;
}

const kernelCache = new Map();

function gaussianKernel(sigmaCells) {
  const sigma = Math.max(0.35, sigmaCells);
  const radius = clamp(Math.ceil(sigma * 3), 1, 10);
  const key = `${radius}:${sigma.toFixed(3)}`;
  if (kernelCache.has(key)) return kernelCache.get(key);
  const weights = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let k = -radius; k <= radius; k += 1) {
    const value = Math.exp(-(k * k) / (2 * sigma * sigma));
    weights[k + radius] = value;
    sum += value;
  }
  for (let i = 0; i < weights.length; i += 1) weights[i] /= sum;
  const kernel = { radius, weights };
  kernelCache.set(key, kernel);
  return kernel;
}

function blurSeparable(input, output, scratch, n, boundary, kernel) {
  const { radius, weights } = kernel;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      let value = 0;
      for (let k = -radius; k <= radius; k += 1) value += weights[k + radius] * sample(input, n, x + k, y, boundary);
      scratch[indexOf(x, y, n)] = value;
    }
  }
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      let value = 0;
      for (let k = -radius; k <= radius; k += 1) value += weights[k + radius] * sample(scratch, n, x, y + k, boundary);
      output[indexOf(x, y, n)] = value;
    }
  }
}

export function surrogateStep(state, configLike, stride = DEFAULT_CONFIG.stride) {
  const config = makeConfig({ ...configLike, stride });
  const { n } = config;
  const tau = config.dt * config.stride;
  const sigmaCells = Math.sqrt(Math.max(1e-12, 2 * config.diffusivity * tau)) / config.dx;
  const kernel = gaussianKernel(sigmaCells);

  const blurredTemp = state._scratch;
  const blurredSource = state._scratch2;
  blurSeparable(state.temperature, blurredTemp, state._buffer, n, config.boundary, kernel);
  blurSeparable(state.source, blurredSource, state._buffer, n, config.boundary, gaussianKernel(Math.max(0.45, sigmaCells * 0.75)));

  const next = state._buffer;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const i = indexOf(x, y, n);
      if (isFixedEdge(config, x, y)) {
        next[i] = 0;
        continue;
      }
      const gamma = config.baseCooling + state.cooling[i];
      const decay = Math.exp(-gamma * tau);
      const sourceResponse = gamma > 1e-8 ? (1 - decay) * blurredSource[i] / gamma : tau * blurredSource[i];
      const operatorPrediction = decay * blurredTemp[i] + sourceResponse;

      const localRhs = 0.18 * config.diffusivity * laplacianAt(state.temperature, n, x, y, config) - gamma * state.temperature[i] + state.source[i];
      const localResidual = state.temperature[i] + tau * localRhs;
      next[i] = 0.88 * operatorPrediction + 0.12 * localResidual;
    }
  }

  const old = state.temperature;
  state.temperature = next;
  state._buffer = old;
  state.time += tau;
  state.solverSteps += config.stride;
  state.macroSteps += 1;
  state.lastMode = "ai";
  return state;
}

export function hybridStep(state, configLike) {
  const config = makeConfig(configLike);
  const dueForCorrection = (state.macroSteps + 1) % config.hybridCorrectionInterval === 0;
  if (dueForCorrection) {
    exactStep(state, config, config.stride);
    state.lastMode = "hybrid-correction";
  } else {
    surrogateStep(state, config, config.stride);
    state.lastMode = "hybrid-ai";
  }
  return state;
}

export function gradientRms(field, configLike) {
  const config = makeConfig(configLike);
  const { n } = config;
  let sum = 0;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dtdx = (sample(field, n, x + 1, y, config.boundary) - sample(field, n, x - 1, y, config.boundary)) / (2 * config.dx);
      const dtdy = (sample(field, n, x, y + 1, config.boundary) - sample(field, n, x, y - 1, config.boundary)) / (2 * config.dx);
      sum += dtdx * dtdx + dtdy * dtdy;
    }
  }
  return Math.sqrt(sum / field.length);
}

export function relativeError(candidate, reference) {
  let diff2 = 0;
  let ref2 = 0;
  for (let i = 0; i < candidate.length; i += 1) {
    const diff = candidate[i] - reference[i];
    diff2 += diff * diff;
    ref2 += reference[i] * reference[i];
  }
  return ref2 > 1e-16 ? Math.sqrt(diff2 / ref2) : 0;
}

export function diagnostics(state, configLike, referenceState = null) {
  const config = makeConfig(configLike);
  const field = state.temperature;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sourcePower = 0;
  let coolingPower = 0;
  for (let i = 0; i < field.length; i += 1) {
    const temp = field[i];
    min = Math.min(min, temp);
    max = Math.max(max, temp);
    sum += temp;
    sourcePower += state.source[i];
    coolingPower += (config.baseCooling + state.cooling[i]) * temp;
  }
  const mean = sum / field.length;
  let variance = 0;
  let hotspotCells = 0;
  const threshold = max > 0 ? 0.8 * max : Infinity;
  for (const value of field) {
    const centered = value - mean;
    variance += centered * centered;
    if (value >= threshold) hotspotCells += 1;
  }
  const error = referenceState ? relativeError(field, referenceState.temperature) : 0;
  return {
    time: state.time,
    solverSteps: state.solverSteps,
    macroSteps: state.macroSteps,
    mean,
    min,
    max,
    absoluteMax: config.ambient + max,
    std: Math.sqrt(variance / field.length),
    gradientRms: gradientRms(field, config),
    hotspotFraction: hotspotCells / field.length,
    sourcePower: sourcePower / field.length,
    coolingPower: coolingPower / field.length,
    netPower: (sourcePower - coolingPower) / field.length,
    relativeError: error,
    lastMode: state.lastMode,
  };
}

export function stepMode(mode, state, configLike) {
  const config = makeConfig(configLike);
  if (mode === "exact") return exactStep(state, config, config.stride);
  if (mode === "ai") return surrogateStep(state, config, config.stride);
  if (mode === "hybrid") return hybridStep(state, config);
  throw new Error(`Unknown mode: ${mode}`);
}

export function exportHistoryCsv(history) {
  const columns = ["time", "max", "mean", "std", "gradientRms", "relativeError", "stepsPerSecond", "mode"];
  const rows = [columns.join(",")];
  for (const item of history) {
    rows.push(columns.map((key) => item[key] ?? "").join(","));
  }
  return rows.join("\n");
}

export function heatColor(value, min, max) {
  const t = clamp((value - min) / Math.max(1e-9, max - min), 0, 1);
  const stops = [
    [0.0, [31, 18, 12]],
    [0.20, [105, 28, 22]],
    [0.45, [188, 55, 36]],
    [0.70, [237, 128, 47]],
    [1.0, [255, 230, 150]],
  ];
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const u = (t - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * u),
        Math.round(c0[1] + (c1[1] - c0[1]) * u),
        Math.round(c0[2] + (c1[2] - c0[2]) * u),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

export function sourceColor(value, max) {
  const t = clamp(value / Math.max(1e-9, max), 0, 1);
  return [
    Math.round(248 - 180 * t),
    Math.round(250 - 120 * t),
    Math.round(252 - 200 * t),
  ];
}
