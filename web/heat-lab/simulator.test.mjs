import assert from "node:assert/strict";

import {
  DEFAULT_CONFIG,
  createGeometry,
  createState,
  diagnostics,
  exactStep,
  makeConfig,
  relativeError,
  surrogateStep,
} from "./simulator.js";

function variance(field) {
  let sum = 0;
  for (const value of field) sum += value;
  const mean = sum / field.length;
  let total = 0;
  for (const value of field) total += (value - mean) ** 2;
  return total / field.length;
}

const config = makeConfig({ ...DEFAULT_CONFIG, n: 32, geometry: "hotspots", sourceScale: 1.2 });
const geometry = createGeometry(config);
assert.equal(geometry.source.length, 32 * 32);
assert.equal(geometry.cooling.length, 32 * 32);
assert.ok(Math.max(...geometry.source) > 0.9);

const state = createState(config);
exactStep(state, config, 4);
const diag = diagnostics(state, config);
assert.ok(Number.isFinite(diag.max));
assert.ok(diag.mean > 0);

const noSource = makeConfig({
  ...DEFAULT_CONFIG,
  n: 32,
  geometry: "hotspots",
  sourceScale: 0,
  baseCooling: 0,
  diffusivity: 0.01,
});
const coolingState = createState(noSource);
for (let i = 0; i < coolingState.temperature.length; i += 1) {
  coolingState.temperature[i] = Math.sin(i * 0.37);
}
const before = variance(coolingState.temperature);
exactStep(coolingState, noSource, 8);
assert.ok(variance(coolingState.temperature) < before);

const aiState = createState(config);
surrogateStep(aiState, config, 8);
assert.ok(Number.isFinite(diagnostics(aiState, config).max));
assert.ok(relativeError(aiState.temperature, state.temperature) >= 0);
