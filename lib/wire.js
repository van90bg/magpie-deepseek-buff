/**
 * Pillar 3: DeepSeek wire protocol invariants.
 *
 * The API rejects a replayed turn where `reasoning_content` sits on a message
 * that carries no `tool_calls`. So the turn's reasoning and its tool calls must
 * end up on ONE assistant message:
 *
 *   ✅ { role:'assistant', content:'', reasoning_content:'…', tool_calls:[…] }
 *   ❌ { role:'assistant', reasoning_content:'…' } + { role:'assistant', tool_calls:[…] }
 *
 * Also handles proxies that leak raw  thinking text into the visible stream
 * instead of native reasoning_content deltas.
 */

const THINK_OPEN = '<' + 'think>';
const THINK_CLOSE = '<' + '/think>';

/**
 * Split a streamed/complete text into reasoning + visible content.
 * A dangling open tag stays reasoning; a dangling close tag is dropped.
 *
 * @returns {{ reasoning: string, content: string }}
 */
export function splitThinkTags(text) {
  if (typeof text !== 'string' || text === '') return { reasoning: '', content: '' };

  let reasoning = '';
  let content = '';
  let cursor = 0;

  while (cursor < text.length) {
    const open = text.indexOf(THINK_OPEN, cursor);
    if (open === -1) {
      content += text.slice(cursor);
      break;
    }
    content += text.slice(cursor, open);
    const close = text.indexOf(THINK_CLOSE, open + THINK_OPEN.length);
    if (close === -1) {
      reasoning += text.slice(open + THINK_OPEN.length);
      break;
    }
    reasoning += text.slice(open + THINK_OPEN.length, close);
    cursor = close + THINK_CLOSE.length;
  }

  // A stray closer in visible text (proxy artefacts) is not content.
  content = content.split(THINK_CLOSE).join('');

  return { reasoning, content };
}

/**
 * Streaming variant: feed deltas, get back the deltas to emit. Keeps a small
 * carry buffer so a tag split across two deltas is still recognised.
 */
export class ThinkTagStreamFilter {
  constructor() {
    this.mode = 'content'; // 'content' | 'reasoning'
    this.carry = '';
  }

  /**
   * @param {string} delta
   * @returns {{ reasoning: string, content: string }}
   */
  push(delta) {
    if (typeof delta !== 'string' || delta === '') return { reasoning: '', content: '' };
    const buffer = this.carry + delta;
    const out = { reasoning: '', content: '' };
    let cursor = 0;

    while (cursor < buffer.length) {
      if (this.mode === 'content') {
        const open = buffer.indexOf(THINK_OPEN, cursor);
        if (open === -1) {
          // Hold back a possible partial tag at the tail.
          const safeEnd = Math.max(cursor, buffer.length - (THINK_OPEN.length - 1));
          out.content += buffer.slice(cursor, safeEnd);
          this.carry = buffer.slice(safeEnd);
          return out;
        }
        out.content += buffer.slice(cursor, open);
        this.mode = 'reasoning';
        cursor = open + THINK_OPEN.length;
      } else {
        const close = buffer.indexOf(THINK_CLOSE, cursor);
        if (close === -1) {
          const safeEnd = Math.max(cursor, buffer.length - (THINK_CLOSE.length - 1));
          out.reasoning += buffer.slice(cursor, safeEnd);
          this.carry = buffer.slice(safeEnd);
          return out;
        }
        out.reasoning += buffer.slice(cursor, close);
        this.mode = 'content';
        cursor = close + THINK_CLOSE.length;
      }
    }

    this.carry = '';
    return out;
  }

  /** Flush whatever is left when the stream ends. */
  flush() {
    const rest = this.carry;
    this.carry = '';
    if (rest === '') return { reasoning: '', content: '' };
    return this.mode === 'reasoning' ? { reasoning: rest, content: '' } : { reasoning: '', content: rest };
  }
}

/**
 * Merge consecutive assistant messages so each run of reasoning ends up on the
 * message that carries the tool calls. Non-assistant messages are untouched and
 * never merged with each other.
 *
 * @param {any[]} messages
 * @returns {{ messages: any[], merged: number }}
 */
export function mergeAssistantMessages(messages) {
  if (!Array.isArray(messages)) return { messages: [], merged: 0 };

  const out = [];
  let merged = 0;

  for (const message of messages) {
    if (!message || typeof message !== 'object') {
      out.push(message);
      continue;
    }
    if (message.role !== 'assistant') {
      out.push(message);
      continue;
    }

    const prev = out[out.length - 1];
    if (prev && prev.role === 'assistant') {
      const toolCalls = [...(prev.tool_calls ?? []), ...(message.tool_calls ?? [])];
      const reasoning = message.reasoning_content || prev.reasoning_content || null;
      out[out.length - 1] = {
        ...prev,
        content: [prev.content, message.content].filter((c) => typeof c === 'string' && c !== '').join(''),
        reasoning_content: reasoning,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      };
      merged++;
      continue;
    }

    out.push({ ...message });
  }

  return { messages: out, merged };
}

/**
 * Normalise one assistant turn before it is replayed to the API: think tags are
 * lifted into reasoning_content, empty content is filled, and a reasoning-only
 * message is only kept when the next message is not an assistant tool carrier.
 *
 * @returns {{ message: any, changed: boolean }}
 */
export function normalizeAssistantMessage(message) {
  if (!message || message.role !== 'assistant') return { message, changed: false };

  let changed = false;
  let reasoning = message.reasoning_content ?? '';
  let content = message.content ?? '';

  const split = splitThinkTags(content);
  if (split.reasoning) {
    reasoning = reasoning ? `${reasoning}${split.reasoning}` : split.reasoning;
    content = split.content;
    changed = true;
  }

  const hasToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
  const normalized = { ...message };
  if (reasoning) normalized.reasoning_content = reasoning;
  if (typeof content === 'string') normalized.content = content;
  else if (hasToolCalls) normalized.content = '';
  if (hasToolCalls) normalized.tool_calls = message.tool_calls;

  if (
    normalized.content !== message.content ||
    normalized.reasoning_content !== message.reasoning_content
  ) {
    changed = true;
  }

  return { message: normalized, changed };
}

/**
 * Full wire pass: normalise, drop reasoning-only messages that cannot be
 * replayed, then merge assistant runs.
 */
export function prepareWireMessages(messages) {
  if (!Array.isArray(messages)) return { messages: [], repaired: [] };
  const repaired = new Set();

  const normalised = messages.map((m) => {
    const { message, changed } = normalizeAssistantMessage(m);
    if (changed) repaired.add('think-tags-lifted');
    return message;
  });

  // A reasoning-only assistant message with no tool calls and no content is
  // not replayable; merging below folds it into its neighbour when possible.
  const { messages: mergedMessages, merged } = mergeAssistantMessages(normalised);
  if (merged > 0) repaired.add('assistant-messages-merged');

  return { messages: mergedMessages, repaired: [...repaired] };
}
