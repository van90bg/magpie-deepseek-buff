/**
 * Public surface: one function that applies the six pillars to a request.
 *
 * Every pass is independently switchable and every pass reports what it did, so
 * a caller can assert on `report.repairs` instead of guessing whether the buff
 * actually engaged.
 */

import { resolveConfig } from './config.js';
import {
  BATCHING_SYSTEM_PROMPT,
  BATCHING_STEP_PROMPT,
  NIT_PICK_SYSTEM_PROMPT,
  reasoningDirective,
} from './prompts.js';
import { repairToolCalls } from './repair.js';
import { prepareWireMessages } from './wire.js';
import { measureContext, planCompaction, trimContext } from './compaction.js';

export { resolveConfig, mapReasoningEffort, DEFAULTS, REASONING_EFFORTS } from './config.js';
export {
  BATCHING_SYSTEM_PROMPT,
  BATCHING_STEP_PROMPT,
  NIT_PICK_SYSTEM_PROMPT,
  BATCHING_INVARIANT,
  reasoningDirective,
} from './prompts.js';
export {
  repairJsonText,
  repairToolJson,
  parseToolCallArguments,
  repairToolCalls,
  canonicalToolName,
  TOOL_NAME_ALIASES,
} from './repair.js';
export {
  splitThinkTags,
  ThinkTagStreamFilter,
  mergeAssistantMessages,
  normalizeAssistantMessage,
  prepareWireMessages,
} from './wire.js';
export {
  estimateTokens,
  measureContext,
  planCompaction,
  trimContext,
} from './compaction.js';

/**
 * Apply the buff.
 *
 * @param {{
 *   systemPrompt?: string,
 *   messages?: any[],
 *   reasoningEffort?: string,
 *   options?: { nitPickReviewer?: boolean, [k: string]: any }
 * }} request
 * @param {object} [configInput]
 * @returns {{
 *   systemPrompt: string,
 *   messages: any[],
 *   stepPrompt: string,
 *   config: any,
 *   report: {
 *     applied: string[],
 *     repairs: string[],
 *     tokensBefore: number,
 *     tokensAfter: number,
 *     compaction: ReturnType<typeof planCompaction> | null,
 *     droppedTokens: number
 *   }
 * }}
 */
export function applyDeepSeekBuff(request = {}, configInput = {}) {
  const config = resolveConfig({ ...configInput, ...(request.options ?? {}) });
  const applied = [];
  const repairs = new Set();

  let systemPrompt = typeof request.systemPrompt === 'string' ? request.systemPrompt : '';
  let messages = Array.isArray(request.messages) ? request.messages.map((m) => ({ ...m })) : [];

  const tokensBefore = measureContext(messages).tokens;

  // Pillar 1a: batching architecture on the system prompt.
  if (config.batching && !systemPrompt.includes('## Work in batches')) {
    systemPrompt = systemPrompt ? `${systemPrompt}\n\n${BATCHING_SYSTEM_PROMPT}` : BATCHING_SYSTEM_PROMPT;
    applied.push('batching-system-prompt');
  }

  // Pillar 6: optional reviewer role.
  if (config.nitPickReviewer && !systemPrompt.includes('DO NOT CALL ANY TOOLS')) {
    systemPrompt = systemPrompt ? `${systemPrompt}\n\n${NIT_PICK_SYSTEM_PROMPT}` : NIT_PICK_SYSTEM_PROMPT;
    applied.push('nit-pick-reviewer');
  }

  // Pillar 5: reasoning ladder (config.reasoningEffort is already mapped).
  const directive = reasoningDirective(config.reasoningEffort);
  if (directive) {
    systemPrompt = `${systemPrompt}\n\n${directive}`;
    applied.push(`reasoning-${config.reasoningEffort}`);
  }

  // Pillar 3: wire protocol before anything else touches the messages.
  if (config.wireProtocol) {
    const wire = prepareWireMessages(messages);
    messages = wire.messages;
    for (const r of wire.repaired) repairs.add(r);
    if (wire.repaired.length > 0) applied.push('wire-protocol');
  }

  // Pillar 2: tool-call names + argument JSON.
  if (config.jsonRepair) {
    let touched = false;
    messages = messages.map((message) => {
      if (!Array.isArray(message.tool_calls) || message.tool_calls.length === 0) return message;
      const { calls, repairs: r } = repairToolCalls(message.tool_calls);
      for (const rule of r) repairs.add(rule);
      if (r.length > 0) touched = true;
      return { ...message, tool_calls: calls };
    });
    if (touched) applied.push('tool-call-repair');
  }

  // Pillar 4: cache-aware compaction decision.
  let compaction = null;
  let droppedTokens = 0;
  if (config.compaction) {
    compaction = planCompaction(messages, {
      config,
      lastActivityAt: request.lastActivityAt,
      now: request.now,
    });
    if (compaction.shouldCompact) {
      const target = Math.floor(config.contextCeiling * config.contextThreshold * 0.5);
      const trimmed = trimContext(messages, target);
      messages = trimmed.kept;
      droppedTokens = trimmed.droppedTokens;
      applied.push(`compaction:${compaction.reason}`);
    }
  }

  const tokensAfter = measureContext(messages).tokens;

  return {
    systemPrompt,
    messages,
    stepPrompt: config.batching ? BATCHING_STEP_PROMPT : '',
    config,
    report: {
      applied,
      repairs: [...repairs],
      tokensBefore,
      tokensAfter,
      compaction,
      droppedTokens,
    },
  };
}

export default applyDeepSeekBuff;