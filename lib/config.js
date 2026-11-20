/**
 * Configuration + defaults. Plain object merge, no schema dependency:
 * a missing field never crashes a caller, it just takes the default.
 */

export const REASONING_EFFORTS = ['low', 'medium', 'high', 'max'];

/** DeepSeek wire limits (see skill doc Pillar 4). */
export const DEFAULTS = Object.freeze({
  batching: true,
  jsonRepair: true,
  wireProtocol: true,
  compaction: true,
  nitPickReviewer: false,

  /** DeepSeek prefix cache TTL ≈ 15 min. After a longer idle gap, prefill is cold. */
  cacheExpiryMs: 15 * 60 * 1000,
  /** Compact history after an idle gap only when it is at least this big. */
  cacheExpiryMinTokens: 40_000,
  /** Hard context ceiling for DeepSeek models (tokens). */
  contextCeiling: 312_500,
  /** Compact when usage crosses this fraction of the ceiling. */
  contextThreshold: 0.8,

  /** Reasoning ladder. 'medium' is mapped up to 'high' — never clamped down. */
  reasoningEffort: 'high',
});

const COERCIBLE_BOOLEANS = ['batching', 'jsonRepair', 'wireProtocol', 'compaction', 'nitPickReviewer'];
const NUMERIC_KEYS = ['cacheExpiryMs', 'cacheExpiryMinTokens', 'contextCeiling', 'contextThreshold'];

/**
 * Build a config from user input, dropping unknown/invalid values instead of
 * throwing, so a bad option in one call site cannot break a whole pipeline.
 */
export function resolveConfig(input = {}) {
  const cfg = { ...DEFAULTS };
  if (!input || typeof input !== 'object') return cfg;

  for (const key of COERCIBLE_BOOLEANS) {
    if (typeof input[key] === 'boolean') cfg[key] = input[key];
  }
  for (const key of NUMERIC_KEYS) {
    const v = input[key];
    if (typeof v === 'number' && Number.isFinite(v) && v > 0) cfg[key] = v;
  }
  // Reasoning effort goes through the ladder so 'medium' can never survive.
  if (input.reasoningEffort != null) {
    cfg.reasoningEffort = mapReasoningEffort(input.reasoningEffort);
  }
  return cfg;
}

/**
 * Pillar 5: 'medium' → 'high'. Never clamp down; an unknown tier → 'high'.
 */
export function mapReasoningEffort(requested) {
  const tier = typeof requested === 'string' ? requested.trim().toLowerCase() : '';
  if (tier === 'max') return 'max';
  if (tier === 'high' || tier === 'medium') return 'high';
  if (tier === 'low') return 'low';
  return 'high';
}
