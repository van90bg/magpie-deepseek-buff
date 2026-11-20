/**
 * Pillar 4: cache-aware compaction.
 *
 * DeepSeek providers prefix-cache aggressively with a ~15 minute TTL. A cold
 * prefill costs ~40x a cached read, so history that sat idle longer than the
 * TTL is worth compacting *even when it is small by absolute standards* — that
 * is what the 40k-token floor encodes: below it, re-prefill is cheap enough
 * that compaction would cost more than it saves.
 *
 * This module decides and trims; it does not summarise. Summarising needs a
 * model call, which belongs to the caller's pipeline.
 */

/** Rough token estimate. UTF-8 bytes / 4: Vietnamese diacritics are 2-3 bytes
 * per character, so counting JS `.length` (UTF-16 units) under-reports badly. */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text === '') return 0;
  return Math.ceil(Buffer.byteLength(text, 'utf8') / 4);
}

function messageTokens(message) {
  if (!message || typeof message !== 'object') return 0;
  let total = 4; // role + envelope overhead
  if (typeof message.content === 'string') total += estimateTokens(message.content);
  if (typeof message.reasoning_content === 'string') total += estimateTokens(message.reasoning_content);
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      total += estimateTokens(JSON.stringify(call ?? {}));
    }
  }
  return total;
}

/** @returns {{ tokens: number, perMessage: number[], systemTokens: number }} */
export function measureContext(messages) {
  const perMessage = Array.isArray(messages) ? messages.map(messageTokens) : [];
  const tokens = perMessage.reduce((a, b) => a + b, 0);
  const systemTokens = Array.isArray(messages)
    ? messages.filter((m) => m?.role === 'system').reduce((a, m) => a + messageTokens(m), 0)
    : 0;
  return { tokens, perMessage, systemTokens };
}

/**
 * Decide whether the history should be compacted before the next turn.
 *
 * @param {any[]} messages
 * @param {{ tokens?: number, lastActivityAt?: number, now?: number, config: any }} state
 * @returns {{ shouldCompact: boolean, reason: string|null, tokens: number, idleMs: number }}
 */
export function planCompaction(messages, state = {}) {
  const { config } = state;
  const now = state.now ?? Date.now();
  const tokens = state.tokens ?? measureContext(messages).tokens;
  const idleMs =
    typeof state.lastActivityAt === 'number' ? Math.max(0, now - state.lastActivityAt) : 0;

  const ceiling = config?.contextCeiling ?? 312_500;
  const threshold = config?.contextThreshold ?? 0.8;
  const cacheExpiryMs = config?.cacheExpiryMs ?? 15 * 60 * 1000;
  const cacheFloor = config?.cacheExpiryMinTokens ?? 40_000;

  if (tokens >= ceiling * threshold) {
    return { shouldCompact: true, reason: 'context-threshold', tokens, idleMs };
  }
  if (idleMs >= cacheExpiryMs && tokens >= cacheFloor) {
    return { shouldCompact: true, reason: 'cache-expired', tokens, idleMs };
  }
  return { shouldCompact: false, reason: null, tokens, idleMs };
}

/**
 * Trim history down to a budget: keep every system message, keep the most
 * recent messages, and return the dropped prefix so the caller can summarise it
 * into a single message instead of losing it.
 *
 * @returns {{ kept: any[], dropped: any[], droppedTokens: number, keptTokens: number }}
 */
export function trimContext(messages, targetTokens, options = {}) {
  const list = Array.isArray(messages) ? messages : [];
  const keepSystem = options.keepSystem !== false;
  const minKeep = options.minKeepMessages ?? 4;

  const systems = keepSystem ? list.filter((m) => m?.role === 'system') : [];
  const rest = keepSystem ? list.filter((m) => m?.role !== 'system') : [...list];

  const systemTokens = systems.reduce((a, m) => a + messageTokens(m), 0);
  let budget = Math.max(0, targetTokens - systemTokens);

  const kept = [];
  for (let i = rest.length - 1; i >= 0; i--) {
    const cost = messageTokens(rest[i]);
    if (kept.length >= minKeep && cost > budget) break;
    kept.unshift(rest[i]);
    budget -= cost;
  }

  const keptIds = new Set(kept);
  const dropped = rest.filter((m) => !keptIds.has(m));

  return {
    kept: [...systems, ...kept],
    dropped,
    droppedTokens: dropped.reduce((a, m) => a + messageTokens(m), 0),
    keptTokens: systemTokens + kept.reduce((a, m) => a + messageTokens(m), 0),
  };
}