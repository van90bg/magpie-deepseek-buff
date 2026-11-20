# magpie-deepseek-buff

A [magpie](https://usemagpie.ai) gateway middleware that gives every **DeepSeek**
turn the things DeepSeek needs and an agent does not do for it.

It applies to a turn DeepSeek answers, and to nothing else:

- a request whose `body.model` names DeepSeek (`auto/deepseek-v4-flash`,
  `freepi/deepseek-v4.1-flash`, `deploy/deepseek-ai/deepseek-v4.1-flash-1017`);
- a request on a routing group whose name says nothing (`group/se-coder`),
  decided **per try** on `ctx.provider.model` — the member actually being
  asked. That is the reason this is middleware and not a fixed system prompt:
  `group/se-coder` can answer on DeepSeek or on Particle-1, and only the
  DeepSeek tries should be buffed.

A provider whose own name contains it (`deepseekgateway/llama-3`) is not a
DeepSeek model, and is left alone.

## Install

```sh
magpie plugin add github:van90bg/magpie-deepseek-buff
magpie plugin options magpie-deepseek-buff '{"batching": true}'
```

In the app: **Plugins › Discover › Gateway middleware**, then its **Options**
button.

## What it does

| Pass | Effect | Default |
|---|---|---|
| `batching` | Appends a *Work in batches* section to the system prompt — Chat's first system/developer message, Anthropic's `system`, Responses' `instructions` — when it is not already there. Appended, never prepended, so the agent's own prompt keeps its cache prefix. | on |
| `heightenReasoning` | Raises `reasoning_effort` `minimal`/`low`/`medium` to `high`. It never invents the field: a provider that rejects an unknown one would fail a request that used to work. | on |
| `repairHistory` | Lifts a `<think>` block a proxy left in an assistant message's text into `reasoning_content`, and repairs tool-call argument JSON DeepSeek writes broken: an unescaped quote, a raw newline inside a string, an invalid escape, an unclosed array, and the classic `{"path": "/etc/hosts}` where the container closes before the string does. A repair that cannot parse leaves the arguments byte for byte. | on |
| `stripReplyThink` | Takes a **leading** `<think>` block out of the reply's text, whole or streamed, so an agent does not keep it in its history and send it back every turn. A `<think>` later in the reply is code the model wrote, an example, and stays. | on |
| `trimToTokens` | Trims history to a token budget. Off by default, because dropping history changes what the agent still knows and that is the operator's call, not a plugin's. | `0` (off) |

Streaming is stripped for Chat (`choices[].delta.content`) and for Responses
(`response.output_text.delta`, reconciled at the part's `…done` event). An
**Anthropic** stream is left whole on purpose: a `text_delta` stream has no
event that carries the block's full text back, so text held back while a tag
was still possible could be lost if the stream stopped mid-tag. Losing a word
is worse than leaking a tag; a non-streaming Anthropic reply is still stripped.

Nothing else about the request changes: not the tools, not the model, not
streaming. Magpie rolls a request back when a middleware changes any of the
three, so this one does not.

## Options

```json
{
  "batching": true,
  "heightenReasoning": true,
  "repairHistory": true,
  "stripReplyThink": true,
  "trimToTokens": 0
}
```

## Test

```sh
npm test
```

25 tests, `node:test`, no dependencies. They cover which turns are buffed
(including a group decided per try, and a provider name that only looks like
DeepSeek), each pass on its own, each one turned off, and the streamed strip
including the case where a stream ends while its first characters were still a
possible tag.

## Licence

MIT.
