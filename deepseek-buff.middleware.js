// deepseek-buff — a magpie gateway middleware that gives a DeepSeek turn the
// things DeepSeek needs and an agent does not do for it.
//
//   {"batching": true, "heightenReasoning": true, "repairHistory": true,
//    "stripReplyThink": true, "trimContext": false}
//
// It applies to a turn DeepSeek answers, and to nothing else:
//   - a request naming DeepSeek in body.model is one, and
//   - a request on a routing group that doesn't name it is decided on each
//     try, where ctx.provider.model says which member is being asked.
// That second point is the whole reason this is middleware and not a fixed
// system prompt: `group/se-coder` can answer on DeepSeek or on Particle-1,
// and only the DeepSeek tries should be buffed.
//
// What it does on such a turn:
//
//   1. batching — appends deepseek-buff's batching section to the system
//      prompt when it isn't already there. Appended, never prepended, so the
//      agent's own prompt keeps the cache prefix. An agent that already
//      carries the section (all of them do, which is why this is a guard and
//      not a duplicate) is left alone.
//   2. heightenReasoning — raises body.reasoning_effort low/medium/minimal to
//      high. It only raises a value the request already has; it never invents
//      the field, because a provider that rejects an unknown field would fail
//      a request that used to work.
//   3. repairHistory — lifts  thinking blocks a proxy left in an assistant
//      message's text into reasoning_content, and repairs tool-call argument
//      JSON that DeepSeek writes broken (unescaped quotes, raw newlines, an
//      unclosed array). Nothing is dropped: a repair that can't parse leaves
//      the arguments byte for byte.
//   4. stripReplyThink — takes a leading  thinking block out of the reply's
//      text, whole or streamed, so an agent doesn't keep it in its history and
//      send it back every turn. The same block, for a Chat reply, is what
//      3 lifts back out of the history.
//
// Context trimming is off by default: dropping history changes what the agent
// still knows, and that is the operator's call, not a plugin's.
//
// The gateway's contract for this file:
//   export function onRequest(body, ctx)  before routing; return a changed
//     body, or undefined to leave it as it was. A change to the tools, the
//     model or streaming rolls the whole request back — this file changes
//     only the system prompt's text, reasoning_effort and message content.
//   export function onProvider(body, ctx) before each try on a member model.
//   export function onResponse(reply, ctx) after the reply is whole.
//   export function onEvent(ev, ctx) for each streamed event.

import { BATCHING_SYSTEM_PROMPT } from "./lib/index.js"
import { repairToolCalls } from "./lib/repair.js"
import { splitThinkTags } from "./lib/wire.js"
import { trimContext } from "./lib/compaction.js"

const BATCHING_HEADING = "## Work in batches"
const DEEPSEEK = /deepseek/i
const RAISE_FROM = new Set(["minimal", "low", "medium"])

// --- what is a DeepSeek turn ---------------------------------------------

// matches: the word anywhere in the model part, after any provider segment, so
// "freepi/deepseek-v4.1-flash" and "auto/deepseek-v4-flash" both say yes while
// a provider whose own name contains it ("deepseekgateway/llama") says no.
function modelIsDeepSeek(id) {
  if (typeof id !== "string" || id === "") return false
  const parts = id.split("/")
  const model = parts.length > 1 ? parts.slice(1).join("/") : parts[0]
  return DEEPSEEK.test(model)
}

function providerModel(ctx) {
  const p = (ctx && ctx.provider) || {}
  const model = typeof p.model === "string" ? p.model : ""
  if (!model) return ""
  return p.id ? p.id + "/" + model : model
}

// buffed is whether this request is being buffed already: what onRequest
// decided, or what onProvider decided for a group that didn't say.
function buffed(ctx) {
  return !!(ctx && ctx.state && ctx.state.deepseekBuff)
}

// stripping is the narrower question the reply-side hooks ask: this turn is a
// buffed DeepSeek turn AND stripReplyThink is on.
function stripping(ctx) {
  return !!(ctx && ctx.state && ctx.state.deepseekBuffStrip)
}

function mark(ctx) {
  if (ctx && ctx.state) ctx.state.deepseekBuff = true
}

// --- 1. batching + 2. reasoning ------------------------------------------

export function onRequest(body, ctx) {
  if (!body || typeof body !== "object") return
  const model = typeof body.model === "string" ? body.model : String((ctx && ctx.model) || "")
  if (!modelIsDeepSeek(model)) return
  mark(ctx)
  return buff(body, ctx)
}

// A group's request names the group; the member being asked is known only
// here. Each try starts from the body onRequest left, so a try on a member
// that isn't DeepSeek keeps its own body and none of the buff.
export function onProvider(body, ctx) {
  if (!body || typeof body !== "object") return
  if (buffed(ctx)) return
  if (!modelIsDeepSeek(providerModel(ctx))) return
  mark(ctx)
  return buff(body, ctx)
}

function option(ctx, name, fallback) {
  const o = (ctx && ctx.options) || {}
  return o[name] === undefined ? fallback : o[name]
}

function buff(body, ctx) {
  let next = body

  if (option(ctx, "batching", true) === true) next = withBatching(next, ctx)
  if (option(ctx, "heightenReasoning", true) === true) next = withReasoning(next)
  if (option(ctx, "repairHistory", true) === true) next = withRepairedHistory(next)
  if (option(ctx, "stripReplyThink", true) !== false && ctx && ctx.state) ctx.state.deepseekBuffStrip = true

  const trimming = option(ctx, "trimContext", false) === true
  if (trimming) next = withTrimmedContext(next, ctx)

  return next === body ? undefined : next
}

// --- system prompt placement, by protocol --------------------------------

function textOfPart(p) {
  if (typeof p === "string") return p
  if (p && typeof p.text === "string") return p.text
  return ""
}

function hasBatching(system) {
  if (typeof system === "string") return system.includes(BATCHING_HEADING)
  if (Array.isArray(system)) return system.some((p) => hasBatching(textOfPart(p)))
  if (system && typeof system.text === "string") return system.text.includes(BATCHING_HEADING)
  return false
}

function withBatching(body, ctx) {
  const protocol = (ctx && ctx.protocol) || detectProtocol(body)

  if (protocol === "anthropic") {
    if (hasBatching(body.system)) return body
    if (typeof body.system === "string") {
      return { ...body, system: body.system + "\n\n" + BATCHING_SYSTEM_PROMPT }
    }
    const blocks = Array.isArray(body.system) ? body.system : []
    if (blocks.length === 0) {
      return { ...body, system: [{ type: "text", text: BATCHING_SYSTEM_PROMPT }] }
    }
    return { ...body, system: [...blocks, { type: "text", text: BATCHING_SYSTEM_PROMPT }] }
  }

  if (protocol === "responses") {
    if (hasBatching(body.instructions)) return body
    if (typeof body.instructions === "string" && body.instructions) {
      return { ...body, instructions: body.instructions + "\n\n" + BATCHING_SYSTEM_PROMPT }
    }
    // An agent's own instructions win over a plugin's: without any, nothing is
    // added, because a Responses turn with no instructions is a different
    // agent shape (a bare completion) that this doesn't need to change.
    return body
  }

  // chat: the system prompt is the first system/developer message, if any.
  const messages = Array.isArray(body.messages) ? body.messages : []
  const at = messages.findIndex((m) => m && (m.role === "system" || m.role === "developer"))
  if (at === -1) return body
  const first = messages[at]
  if (hasBatching(first.content)) return body
  if (typeof first.content === "string") {
    const content = first.content + "\n\n" + BATCHING_SYSTEM_PROMPT
    const copy = messages.slice()
    copy[at] = { ...first, content }
    return { ...body, messages: copy }
  }
  if (Array.isArray(first.content)) {
    const copy = messages.slice()
    copy[at] = { ...first, content: [...first.content, { type: "text", text: BATCHING_SYSTEM_PROMPT }] }
    return { ...body, messages: copy }
  }
  return body
}

function detectProtocol(body) {
  if (Array.isArray(body.messages)) return "chat"
  if (typeof body.instructions === "string" || Array.isArray(body.input)) return "responses"
  if (body.system !== undefined || body.anthropic_version) return "anthropic"
  return "chat"
}

// --- reasoning effort ----------------------------------------------------

function withReasoning(body) {
  const effort = typeof body.reasoning_effort === "string" ? body.reasoning_effort.toLowerCase() : ""
  if (!RAISE_FROM.has(effort)) return body
  return { ...body, reasoning_effort: "high" }
}

// --- 3. history repair ---------------------------------------------------

function liftThink(message) {
  if (!message || message.role !== "assistant") return message
  let changed = false
  let next = message

  const content = message.content
  const text = typeof content === "string" ? content : ""
  if (text.includes("<" + "think")) {
    const { reasoning, content: visible } = splitThinkTags(text)
    const had = typeof message.reasoning_content === "string" ? message.reasoning_content : ""
    const merged = [had, reasoning].filter(Boolean).join("\n").trim()
    if (visible !== text || merged !== had) {
      next = { ...next, content: visible }
      if (merged) next.reasoning_content = merged
      changed = true
    }
  }

  if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
    const { calls, repairs } = repairToolCalls(message.tool_calls)
    if (repairs.length) {
      next = { ...next, tool_calls: calls }
      changed = true
    }
  }

  return changed ? next : message
}

function withRepairedHistory(body) {
  const messages = Array.isArray(body.messages) ? body.messages : []
  if (messages.length === 0) return body
  let changed = false
  const next = messages.map((m) => {
    const fixed = liftThink(m)
    if (fixed !== m) changed = true
    return fixed
  })
  return changed ? { ...body, messages: next } : body
}

// --- context trimming (opt-in) -------------------------------------------

function withTrimmedContext(body, ctx) {
  const messages = Array.isArray(body.messages) ? body.messages : []
  if (messages.length === 0) return body
  const target = Number(option(ctx, "trimToTokens", 0)) || 0
  if (!target) return body
  const trimmed = trimContext(messages, target, { keepSystem: true })
  return trimmed === messages ? body : { ...body, messages: trimmed }
}

// --- 4. reply: a leading think block is not text -------------------------

const OPEN = "<" + "think" + ">"
const CLOSE = "<" + "/think" + ">"

// stripLeadingText takes a leading block out of a whole reply's text and
// returns it as reasoning. Only a block that opens the text is taken: a
//  thinking later on is code the model wrote, an example, and stays.
function stripLeadingText(text) {
  if (typeof text !== "string" || text === "") return { content: text, reasoning: "" }
  const open = text.indexOf(OPEN)
  const { reasoning, content } = splitThinkTags(text)
  const opensText = open === 0 || (open > 0 && text.slice(0, open).trim() === "")
  if (!reasoning || !opensText) return { content: text, reasoning: "" }
  return { content, reasoning }
}

// LeadingThinkFilter strips one leading thinking block from a streamed reply.
//
// It holds text back ONLY while the reply's opening characters are still a
// possible prefix of the open tag. The moment they are not, everything held
// so far is text and is emitted at once, and the filter never holds anything
// again. A stream that ends while undecided gives its hold-back back, because
// that is text; a stream that ends inside the block drops it, because that is
// thinking. Losing a word is worse than leaking a tag, so every uncertain
// byte ends up as text.
//
//   0 undecided · 1 inside the block · 2 done (the rest is text)
class LeadingThinkFilter {
  constructor() {
    this.mode = 0
  }

  push(delta) {
    if (typeof delta !== "string" || delta === "") return { content: "", reasoning: "" }
    if (this.mode === 2) return { content: delta, reasoning: "" }
    if (this.mode === 1) return this.inside(delta)
    return this.undecided(delta)
  }

  undecided(delta) {
    // Does the text so far still spell out a prefix of the open tag, once the
    // blank space a model may put first ("\n\n<think>") is set aside?
    this.pending = (this.pending || "") + delta
    const p = this.pending
    const lead = p.length - p.trimStart().length
    const body = p.slice(lead)
    if (body === "") return { content: "", reasoning: "" }
    if (body.length < OPEN.length && OPEN.startsWith(body)) return { content: "", reasoning: "" }
    if (body.startsWith(OPEN)) {
      this.mode = 1
      this.pending = ""
      return this.inside(body.slice(OPEN.length))
    }
    // Not a thinking block: it is all text, blank space included.
    this.mode = 2
    this.pending = ""
    return { content: p, reasoning: "" }
  }

  inside(delta) {
    this.pending = (this.pending || "") + delta
    const p = this.pending
    const close = p.indexOf(CLOSE)
    if (close === -1) {
      // Hold back a possible partial close tag at the tail only.
      const keep = Math.max(0, CLOSE.length - 1)
      const safe = p.length > keep ? p.slice(0, p.length - keep) : ""
      this.pending = p.slice(safe.length)
      return { reasoning: safe, content: "" }
    }
    this.mode = 2
    this.pending = ""
    return { reasoning: p.slice(0, close), content: p.slice(close + CLOSE.length) }
  }

  /** end(): the last of the stream, and the block never closed. */
  end() {
    const rest = this.pending || ""
    this.pending = ""
    if (this.mode === 1) {
      this.mode = 2
      return { content: "", reasoning: rest }
    }
    this.mode = 2
    return { content: rest, reasoning: "" }
  }
}

function filterFor(ctx, key) {
  if (!ctx || !ctx.state) return new LeadingThinkFilter()
  const filters = ctx.state.deepseekBuffFilters || (ctx.state.deepseekBuffFilters = new Map())
  let filter = filters.get(key)
  if (!filter) {
    filter = new LeadingThinkFilter()
    filters.set(key, filter)
  }
  return filter
}

// emit applies one delta: text goes where it belongs, thinking to
// reasoning_content when the reply carries one.
function emit(filter, text, holder) {
  const out = filter.push(text)
  let changed = false
  if (out.reasoning) {
    holder.reasoning += out.reasoning
    changed = true
  }
  if (out.content !== text) {
    holder.content = out.content
    changed = true
  }
  return changed
}

export function onResponse(reply, ctx) {
  if (!reply || typeof reply !== "object") return
  if (!stripping(ctx)) return
  if (typeof ctx.status === "number" && ctx.status >= 400) return
  let changed = false

  if (Array.isArray(reply.choices)) {
    for (const choice of reply.choices) {
      const message = choice && choice.message
      if (!message || typeof message.content !== "string") continue
      const { content, reasoning } = stripLeadingText(message.content)
      if (content === message.content) continue
      message.content = content
      if (reasoning && typeof message.reasoning_content !== "string") {
        message.reasoning_content = reasoning
      }
      changed = true
    }
  }

  if (Array.isArray(reply.content)) {
    for (const block of reply.content) {
      if (!block || block.type !== "text" || typeof block.text !== "string") continue
      const { content } = stripLeadingText(block.text)
      if (content === block.text) continue
      block.text = content
      changed = true
    }
  }

  if (Array.isArray(reply.output)) {
    for (const item of reply.output) {
      if (!item || !Array.isArray(item.content)) continue
      for (const part of item.content) {
        if (!part || typeof part.text !== "string") continue
        const { content } = stripLeadingText(part.text)
        if (content === part.text) continue
        part.text = content
        changed = true
      }
    }
  }

  return changed ? reply : undefined
}

export function onEvent(ev, ctx) {
  if (!ev || typeof ev !== "object") return
  if (!stripping(ctx)) return
  let changed = false

  // Chat: choices[].delta.content, and the finish that flushes a hold-back.
  if (Array.isArray(ev.choices)) {
    for (const choice of ev.choices) {
      const delta = choice && choice.delta
      if (!delta) continue
      const filter = filterFor(ctx, "c" + choice.index)
      const holder = {
        content: typeof delta.content === "string" ? delta.content : "",
        reasoning: typeof delta.reasoning_content === "string" ? delta.reasoning_content : "",
      }
      let touched = false
      if (typeof delta.content === "string" && delta.content !== "") {
        touched = emit(filter, delta.content, holder)
      }
      if (choice.finish_reason) {
        const last = filter.end()
        if (last.content) {
          holder.content += last.content
          touched = true
        }
        if (last.reasoning) {
          holder.reasoning += last.reasoning
          touched = true
        }
      }
      if (!touched) continue
      if (holder.content === "") delete delta.content
      else delta.content = holder.content
      if (holder.reasoning) delta.reasoning_content = holder.reasoning
      changed = true
    }
    return changed ? ev : undefined
  }

  // Responses: response.output_text.delta, with the part's done event as the
  // flush point. The done event carries the whole text, so rewriting it both
  // flushes the hold-back and settles anything the deltas left behind.
  if (ev.type === "response.output_text.delta" && typeof ev.delta === "string") {
    const filter = filterFor(ctx, "r" + ev.item_id + ":" + ev.content_index)
    const holder = { content: ev.delta, reasoning: "" }
    if (emit(filter, ev.delta, holder)) {
      ev.delta = holder.content
      changed = true
    }
    return changed ? ev : undefined
  }
  if ((ev.type === "response.output_text.done" && typeof ev.text === "string") ||
      (ev.type === "response.content_part.done" && ev.part && typeof ev.part.text === "string")) {
    const whole = ev.type === "response.output_text.done" ? ev.text : ev.part.text
    const { content } = stripLeadingText(whole)
    const filter = filterFor(ctx, "r" + ev.item_id + ":" + ev.content_index)
    filter.end()
    if (content === whole) return
    if (ev.type === "response.output_text.done") ev.text = content
    else ev.part.text = content
    return ev
  }

  // Anthropic streaming is left whole on purpose: a text_delta stream gives
  // no event that carries the block's full text back, so a block stripped
  // from the deltas could lose the last few characters if the stream stopped
  // mid-tag. Losing a word is worse than leaking a tag, and a non-streaming
  // Anthropic reply still gets stripped in onResponse.
  return
}
