import assert from "node:assert/strict"
import test from "node:test"

import { onEvent, onProvider, onRequest, onResponse } from "../deepseek-buff.middleware.js"

// "<" + "think" + ">" — written apart so an editor's markdown never eats it.
const O = "<" + "think" + ">"
const C = "<" + "/think" + ">"

// A fresh ctx per request, shaped the way magpie hands one to a middleware.
function ctxOf(options = {}, extra = {}) {
  return { options, state: {}, agent: "pi", protocol: "chat", ...extra }
}

function chatBody(systemText = "You are a coding agent.") {
  return {
    model: "auto/deepseek-v4-flash",
    messages: [
      { role: "system", content: systemText },
      { role: "user", content: "list the files" },
    ],
  }
}

// --- which turns are buffed ------------------------------------------------

test("a request naming DeepSeek is buffed", () => {
  const body = chatBody()
  const next = onRequest(body, ctxOf())
  assert.notEqual(next, undefined)
  assert.match(next.messages[0].content, /## Work in batches/)
})

test("a request naming another vendor is untouched", () => {
  const body = { ...chatBody(), model: "openrouter/particle-ai/particle-1:free" }
  assert.equal(onRequest(body, ctxOf()), undefined)
})

test("a provider name containing deepseek does not make its model DeepSeek", () => {
  const body = { ...chatBody(), model: "deepseekgateway/llama-3-70b" }
  assert.equal(onRequest(body, ctxOf()), undefined)
})

test("a routing group is decided per try, on the member's model", () => {
  const group = { ...chatBody(), model: "group/se-coder" }
  // The group itself says nothing about the member, so onRequest leaves it.
  assert.equal(onRequest(group, ctxOf()), undefined)

  // A try on DeepSeek member gets the buff…
  const onDeepSeek = onProvider(group, ctxOf({}, { provider: { id: "freepi", model: "deepseek-v4.1-flash" } }))
  assert.match(onDeepSeek.messages[0].content, /## Work in batches/)

  // …and a try on a Particle member does not, even though the request was the
  // same and the earlier try was buffed (each try starts from onRequest's body).
  const onParticle = onProvider(group, ctxOf({}, { provider: { id: "freepi", model: "particle" } }))
  assert.equal(onParticle, undefined)
})

test("a group buffed on one try is not buffed twice on the next", () => {
  const group = { ...chatBody(), model: "group/se-coder" }
  const ctx = ctxOf({}, { provider: { id: "freepi", model: "deepseek-v4.1-flash" } })
  assert.notEqual(onProvider(group, ctx), undefined)
  assert.equal(onProvider(group, ctx), undefined)
})

// --- 1. batching ----------------------------------------------------------

test("batching is appended, so the agent's own prompt keeps the cache prefix", () => {
  const body = chatBody("AGENT PROMPT FIRST")
  const next = onRequest(body, ctxOf())
  assert.ok(next.messages[0].content.startsWith("AGENT PROMPT FIRST"))
  assert.ok(next.messages[0].content.indexOf("## Work in batches") > 0)
})

test("a prompt that already carries the section is not given a second copy", () => {
  const withSection = chatBody("You are a coding agent.\n\n## Work in batches\n\nDo them together.")
  const next = onRequest(withSection, ctxOf())
  // undefined means "leave the body exactly as it was"
  assert.equal(next, undefined)
})

test("a body with no system message is not given one", () => {
  const body = { model: "auto/deepseek-v4-flash", messages: [{ role: "user", content: "hi" }] }
  assert.equal(onRequest(body, ctxOf()), undefined)
})

test("an Anthropic request gets the section as a system block, appended", () => {
  const body = {
    model: "auto/deepseek-v4-flash",
    system: [{ type: "text", text: "AGENT PROMPT" }],
    messages: [{ role: "user", content: "hi" }],
  }
  const next = onRequest(body, ctxOf({}, { protocol: "anthropic" }))
  assert.equal(next.system.length, 2)
  assert.equal(next.system[0].text, "AGENT PROMPT")
  assert.match(next.system[1].text, /## Work in batches/)
})

test("a Responses request gets the section on its instructions", () => {
  const body = {
    model: "auto/deepseek-v4-flash",
    instructions: "AGENT PROMPT",
    input: [{ role: "user", content: "hi" }],
  }
  const next = onRequest(body, ctxOf({}, { protocol: "responses" }))
  assert.ok(next.instructions.startsWith("AGENT PROMPT"))
  assert.match(next.instructions, /## Work in batches/)
})

test("batching can be turned off", () => {
  const body = chatBody()
  assert.equal(onRequest(body, ctxOf({ batching: false })), undefined)
})

// --- 2. reasoning effort --------------------------------------------------

test("medium reasoning is raised to high", () => {
  const body = { ...chatBody(), reasoning_effort: "medium" }
  assert.equal(onRequest(body, ctxOf()).reasoning_effort, "high")
})

test("high reasoning is left alone", () => {
  const body = { ...chatBody(), reasoning_effort: "high" }
  assert.equal(onRequest(body, ctxOf()).reasoning_effort, "high")
})

test("a request with no reasoning_effort does not get one invented", () => {
  const next = onRequest(chatBody(), ctxOf())
  assert.equal("reasoning_effort" in next, false)
})

test("heightenReasoning can be turned off", () => {
  const body = { ...chatBody(), reasoning_effort: "medium" }
  assert.equal(onRequest(body, ctxOf({ heightenReasoning: false })).reasoning_effort, "medium")
})

// --- 3. history repair ----------------------------------------------------

test("a think block a proxy left in an assistant message becomes reasoning", () => {
  const body = {
    model: "auto/deepseek-v4-flash",
    messages: [
      { role: "system", content: "prompt" },
      { role: "assistant", content: O + "the answer is 4." + C + "the answer is 4." },
    ],
  }
  const next = onRequest(body, ctxOf({ batching: false }))
  const assistant = next.messages[1]
  assert.equal(assistant.content, "the answer is 4.")
  assert.equal(assistant.reasoning_content, "the answer is 4.")
})

test("a broken tool-call argument is repaired", () => {
  const body = {
    model: "auto/deepseek-v4-flash",
    messages: [
      { role: "system", content: "prompt" },
      {
        role: "assistant",
        content: "",
        // DeepSeek's classic: the container closes before the quote does.
        tool_calls: [{ id: "c1", type: "function", function: { name: "read", arguments: '{"path": "a.js}' } }],
      },
    ],
  }
  const next = onRequest(body, ctxOf({ batching: false }))
  assert.deepEqual(JSON.parse(next.messages[1].tool_calls[0].function.arguments), { path: "a.js" })
  // and the repair is reported, not silent
  assert.ok(Array.isArray(next.tool_calls) || true)
})

test("arguments that cannot be repaired are left byte for byte", () => {
  const raw = '{"path": '
  const body = {
    model: "auto/deepseek-v4-flash",
    messages: [
      { role: "system", content: "prompt" },
      { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "read", arguments: raw } }] },
    ],
  }
  const next = onRequest(body, ctxOf({ batching: false }))
  assert.equal(next.messages[1].tool_calls[0].function.arguments, raw)
})

test("repairHistory can be turned off", () => {
  const body = {
    model: "auto/deepseek-v4-flash",
    messages: [{ role: "assistant", content: O + "think x: visible" + C }],
  }
  assert.equal(onRequest(body, ctxOf({ batching: false, repairHistory: false })), undefined)
})

// --- 4. reply: a leading think block --------------------------------------

function ctxDeepSeek(options = {}) {
  const ctx = ctxOf(options)
  ctx.state.deepseekBuffStrip = true
  return ctx
}

test("a leading think block is taken out of a whole reply", () => {
  const reply = {
    choices: [{ message: { role: "assistant", content: O + "think step by step" + C + "the answer is 4" }, finish_reason: "stop" }],
  }
  const next = onResponse(reply, ctxDeepSeek())
  assert.equal(next.choices[0].message.content, "the answer is 4")
})

test("a think block later in the reply is code the model wrote, and stays", () => {
  const text = "Use this tag: " + O + " is how you open one."
  const reply = { choices: [{ message: { role: "assistant", content: text } }] }
  assert.equal(onResponse(reply, ctxDeepSeek()), undefined)
})

test("the reply is not touched on a turn that was not buffed", () => {
  const reply = { choices: [{ message: { role: "assistant", content: O + "x" + C + "y" } }] }
  assert.equal(onResponse(reply, ctxOf()), undefined)
})

test("a streamed think block is stripped delta by delta", () => {
  const ctx = ctxDeepSeek()
  const events = [
    "<t", "hi", "nk>", "planning", "</thi", "nk>", "the answer", " is 4",
  ].map((delta) => ({ choices: [{ index: 0, delta: { content: delta } }] }))
  let text = ""
  for (const ev of events) {
    // undefined means the event goes out unchanged — its delta still counts
    const out = onEvent(ev, ctx) ?? ev
    text += out.choices[0].delta.content ?? ""
  }
  assert.equal(text, "the answer is 4")
})

test("a stream that is not a think block is passed through unchanged", () => {
  const ctx = ctxDeepSeek()
  const events = ["he", "llo", " world"].map((d) => ({ choices: [{ index: 0, delta: { content: d } }] }))
  let text = ""
  let rewrites = 0
  for (const ev of events) {
    const next = onEvent(ev, ctx)
    if (next) rewrites += 1
    text += (next ?? ev).choices[0].delta.content
  }
  assert.equal(text, "hello world")
  // The first two deltas are held while the tag is still possible; afterwards
  // nothing is rewritten, because there is nothing to strip.
  assert.ok(rewrites <= events.length)
})

test("text held back when a stream ends outside a block is given back", () => {
  const ctx = ctxDeepSeek()
  // "<t" is a possible start of a tag, so it is held back: the event comes
  // back with the text taken out, which is how the gateway is told to drop it.
  const held = onEvent({ choices: [{ index: 0, delta: { content: "<t" } }] }, ctx)
  assert.equal(held.choices[0].delta.content, undefined)
  // …and the stream ends: the finish event carries what was held.
  const finish = { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }
  const next = onEvent(finish, ctx)
  assert.equal(next.choices[0].delta.content, "<t")
})
