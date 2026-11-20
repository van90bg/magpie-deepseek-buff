/**
 * Pillar 2: tool-call JSON repair engine.
 *
 * A character scanner, not a pile of regexes: regexes cannot tell an opening
 * quote of a string from a quote DeepSeek forgot to escape inside one, and
 * that distinction is the whole problem.
 *
 * Repairs, in the order the scanner meets them while inside a string literal:
 *   - raw-control-character : literal newline/tab/CR inside a string → \n \t \r
 *   - invalid-escape        : \x where x is not a legal JSON escape → \\x
 *   - unescaped-quote       : a " whose next meaningful char is not structural
 *                             (`,` `:` `}` `]` or EOF) → \" (JSX/HTML class attrs)
 * Outside a string:
 *   - unclosed-array        : an object opened with `[` closed by `}` → `]}`
 *   - unclosed-container    : EOF with open containers → append the closers
 *
 * Always returns a string; never throws. `repairToolJson` parses and reports.
 */

const VALID_ESCAPES = new Set(['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']);
const STRUCTURAL_AFTER_STRING = new Set([',', ':', '}', ']']);

const RAW_CONTROL_ESCAPES = new Map([
  ['\n', '\\n'],
  ['\r', '\\r'],
  ['\t', '\\t'],
  ['\b', '\\b'],
  ['\f', '\\f'],
]);

function nextMeaningfulChar(text, from) {
  for (let i = from; i < text.length; i++) {
    const c = text[i];
    if (c !== ' ' && c !== '\t' && c !== '\n' && c !== '\r') return c;
  }
  return '';
}

/**
 * @param {string} input malformed JSON text
 * @returns {{ text: string, repairs: string[] }} repaired text + rule names hit
 */
export function repairJsonText(input) {
  if (typeof input !== 'string' || input === '') {
    return { text: typeof input === 'string' ? input : '', repairs: [] };
  }

  const repairs = new Set();
  const stack = []; // '{' or '['
  let out = '';
  let inString = false;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (inString) {
      if (ch === '\\') {
        const escaped = input[i + 1];
        if (escaped === undefined) {
          // Trailing lone backslash: make it a legal escaped backslash.
          out += '\\\\';
          repairs.add('invalid-escape');
          continue;
        }
        if (!VALID_ESCAPES.has(escaped)) {
          // dir\file, \s \d \w in regexes → double the backslash.
          out += '\\\\' + escaped;
          repairs.add('invalid-escape');
          i++;
          continue;
        }
        out += ch + escaped;
        i++;
        continue;
      }

      if (ch === '"') {
        const after = nextMeaningfulChar(input, i + 1);
        if (after === '' || STRUCTURAL_AFTER_STRING.has(after)) {
          inString = false;
          out += ch;
        } else {
          // Interior quote (e.g. className="card") → escape it.
          repairs.add('unescaped-quote');
          out += '\\"';
        }
        continue;
      }

      if (RAW_CONTROL_ESCAPES.has(ch)) {
        repairs.add('raw-control-character');
        out += RAW_CONTROL_ESCAPES.get(ch);
        continue;
      }

      out += ch;
      continue;
    }

    // Outside a string.
    if (ch === '"') {
      inString = true;
      out += ch;
      continue;
    }
    if (ch === '{' || ch === '[') {
      stack.push(ch);
      out += ch;
      continue;
    }
    if (ch === '}' || ch === ']') {
      const top = stack[stack.length - 1];
      if (top === '[' && ch === '}') {
        // DeepSeek closes an array with an object closer: emit the missing
        // ']', then let this same '}' close the container underneath.
        stack.pop();
        out += ']';
        repairs.add('unclosed-array');
        if (stack.length > 0) stack.pop();
        out += '}';
        continue;
      }
      if (top) stack.pop();
      out += ch;
      continue;
    }
    out += ch;
  }

  if (inString) {
    // A string still open at EOF whose last characters are the container's own
    // closers means the model left the closing quote out and wrote the closer
    // inside the string: {"path": "/etc/hosts} is {"path": "/etc/hosts"}.
    // Move only closers that match the innermost open container, one per
    // container, so a value that merely ends with "}" keeps it.
    let moved = '';
    for (;;) {
      const last = out[out.length - 1];
      const top = stack[stack.length - 1];
      if ((last === '}' && top === '{') || (last === ']' && top === '[')) {
        moved = last + moved;
        out = out.slice(0, -1);
        stack.pop();
        continue;
      }
      break;
    }
    out += '"';
    repairs.add('unterminated-string');
    out += moved;
  }
  while (stack.length > 0) {
    out += stack.pop() === '[' ? ']' : '}';
    repairs.add('unclosed-container');
  }

  return { text: out, repairs: [...repairs] };
}

/**
 * Repair, then parse. Never throws: on failure it reports `ok:false` with the
 * original text so the caller can retry the model turn instead of crashing.
 *
 * @returns {{ ok: boolean, value?: any, text: string, repairs: string[], error?: string }}
 */
export function repairToolJson(input) {
  const { text, repairs } = repairJsonText(input);
  try {
    return { ok: true, value: JSON.parse(text), text, repairs };
  } catch (error) {
    return { ok: false, text, repairs, error: error.message };
  }
}

/**
 * Parse a tool call's arguments field, repairing first. DeepSeek sometimes
 * double-encodes arguments as a JSON string, so that is unwrapped too.
 */
export function parseToolCallArguments(raw) {
  if (raw == null) return { ok: true, value: {}, text: '', repairs: [] };
  if (typeof raw === 'object') return { ok: true, value: raw, text: '', repairs: [] };

  const first = repairToolJson(raw);
  if (first.ok && typeof first.value === 'string') {
    const second = repairToolJson(first.value);
    if (second.ok) return second;
  }
  return first;
}

export const TOOL_NAME_ALIASES = Object.freeze({
  run_command: 'bash',
  run_default: 'bash',
  run_exec_command: 'bash',
  terminal_command: 'bash',
  run_terminal_command: 'bash',
  shell: 'bash',
  run_code_search: 'code_search',
  grep: 'code_search',
  search_code: 'code_search',
  run_files: 'read_files',
  read_file: 'read_files',
  cat: 'read_files',
  write_file: 'write_file',
  edit_file: 'edit',
  str_replace: 'edit',
});

/** Map a hallucinated tool name onto a harness tool; unknown names pass through. */
export function canonicalToolName(name) {
  if (typeof name !== 'string') return name;
  const key = name.trim().toLowerCase();
  return TOOL_NAME_ALIASES[key] ?? key;
}

/**
 * Normalise a whole tool-call list: canonical names + repaired arguments.
 * @returns {{ calls: any[], repairs: string[] }}
 */
export function repairToolCalls(calls) {
  const repairs = new Set();
  if (!Array.isArray(calls)) return { calls: [], repairs: [] };

  const out = calls.map((call) => {
    if (!call || typeof call !== 'object') return call;
    const fn = call.function ?? call;
    const name = canonicalToolName(fn?.name);
    if (name !== fn?.name) repairs.add('tool-name-alias');

    const rawArgs = fn?.arguments;
    const parsed = typeof rawArgs === 'string' ? parseToolCallArguments(rawArgs) : { ok: true, value: rawArgs, repairs: [] };
    for (const r of parsed.repairs) repairs.add(r);
    if (!parsed.ok) repairs.add('unparseable-arguments');

    if (call.function) {
      return {
        ...call,
        function: { ...call.function, name, arguments: parsed.ok ? JSON.stringify(parsed.value) : rawArgs },
      };
    }
    return { ...call, name, arguments: parsed.ok ? JSON.stringify(parsed.value) : rawArgs };
  });

  return { calls: out, repairs: [...repairs] };
}
