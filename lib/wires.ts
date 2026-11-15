// The three upstream wires and the two client wires.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages; each
// provider speaks one of OpenAI, Anthropic or Gemini. This file translates
// faithfully in both directions, because the env's client compares what the
// provider echoed against what it sent: a field rewritten or dropped loses
// the request. The fields that must survive untouched are the system prompt,
// the user text (images included), the tool definitions, the stop sequences,
// max_tokens and JSON mode.
//
// The provider's answer is always one plain text, "ANSWER ECHO{...}", with
// stop family 0, so the translations below carry text, usage and the stop
// family through, and map the stop family into the client's wire (stop for
// OpenAI, end_turn for Anthropic). Usage is the provider's own counts, in the
// shape the client's wire reports them; for a streamed OpenAI answer the
// usage rides in a final chunk, which only arrives when the upstream request
// carries stream_options.include_usage.
//
// All timestamps here are fixed constants: nothing in a reply reads the clock,
// so a live pass and its replay print the same bytes.

import { authHeaders, isObject, joinBase, type DirectoryEntry, type Json, type Wire } from './brief.ts';

/** One provider answer, already parsed: its text, its usage, its stop family. */
export interface UpstreamAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  /** 0 stop/end_turn/STOP, 1 length/max_tokens, 2 tool_calls/tool_use, 3 content_filter/refusal. */
  stop: number;
}

/** A provider call that did not answer 2xx, or could not be read. */
export class UpstreamFailure extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isObject(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** One Anthropic content block as an OpenAI content part: text or inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isObject(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isObject(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isObject(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI finish_reason as a stop family number. */
function openAIFamily(finish: unknown): number {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

/** An Anthropic stop_reason as a stop family number. */
function anthropicFamily(reason: unknown): number {
  if (reason === 'max_tokens') return 1;
  if (reason === 'tool_use') return 2;
  if (reason === 'refusal') return 3;
  return 0;
}

/** A Gemini finishReason as a stop family number. */
function geminiFamily(reason: unknown): number {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY') return 3;
  return 0;
}

const familyToOpenAI = (family: number): string => (family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop');
const familyToAnthropic = (family: number): string => (family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn');

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObject(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isObject(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isObject(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isObject(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObject(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isObject(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObject(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A Chat Completions request as an Anthropic Messages request naming `model`:
 * the system message becomes the system prompt, image_url parts become image
 * blocks, tool_calls become tool_use blocks, tool messages become tool_result
 * blocks, and the rest passes through. A JSON-mode request has no Anthropic
 * spelling, so the router never sends one down this path.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObject(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isObject(call) || !isObject(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : toolResultText(content) }] });
      continue;
    }
    // A user message: text and image_url parts become Anthropic blocks.
    const blocks: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isObject(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isObject(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isObject(raw) || !isObject(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isObject(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isObject(choice) && choice.type === 'function' && isObject(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** The max_tokens of an OpenAI-shaped body, from either spelling. */
function openAIMaxTokens(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/** The stop sequences of an OpenAI-shaped body, as an array. */
function openAIStops(body: Json): string[] {
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (typeof body.stop === 'string') return [body.stop];
  return [];
}

/** The stop sequences of an Anthropic-shaped body, as an array. */
function anthropicStops(body: Json): string[] {
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/** OpenAI tools as Gemini function declarations. */
function geminiToolsFromOpenAI(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const decls = body.tools.flatMap((raw) => {
    if (!isObject(raw) || !isObject(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isObject(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : null;
}

/** Anthropic tools as Gemini function declarations. */
function geminiToolsFromAnthropic(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const decls = body.tools.flatMap((raw) => {
    if (!isObject(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isObject(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : null;
}

/** A tool_choice as a Gemini functionCallingConfig, or null when it says nothing. */
function geminiToolChoice(choice: unknown): Json | null {
  if (choice === 'auto' || (isObject(choice) && choice.type === 'auto')) return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice === 'required' || (isObject(choice) && choice.type === 'any')) return { functionCallingConfig: { mode: 'ANY' } };
  if (choice === 'none' || (isObject(choice) && choice.type === 'none')) return { functionCallingConfig: { mode: 'NONE' } };
  if (isObject(choice) && choice.type === 'function' && isObject(choice.function) && typeof choice.function.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  if (isObject(choice) && choice.type === 'tool' && typeof choice.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part: text, or inline image data. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isObject(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part, or null for tool blocks (handled by role). */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isObject(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/**
 * A Chat Completions request as a Gemini generateContent body for `model`:
 * the system message becomes the system instruction, messages become contents
 * (assistant turns are the model role), tools become function declarations,
 * and max_tokens, stops, temperature and JSON mode land in generationConfig.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObject(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isObject(call) || !isObject(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isObject(part)) continue;
        const p = geminiPartOfOpenAI(part);
        if (p !== null) parts.push(p);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents, model };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsFromOpenAI(body);
  if (tools !== null) out.tools = tools;
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isObject(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = geminiToolChoice(body.tool_choice);
  if (choice !== null) out.toolConfig = choice;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body for `model`:
 * the system prompt becomes the system instruction, user turns become user
 * contents, assistant turns become model contents, tool_use blocks become
 * functionCall parts and tool_result blocks become function responses.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isObject(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isObject(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isObject(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user turn: text and image parts, plus any tool_result blocks as a
    // function-role turn of their own.
    const parts: Json[] = [];
    const responses: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isObject(block)) continue;
        if (block.type === 'tool_result') {
          responses.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
          continue;
        }
        const p = geminiPartOfAnthropic(block);
        if (p !== null) parts.push(p);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (responses.length > 0) contents.push({ role: 'function', parts: responses });
  }

  const out: Json = { contents, model };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsFromAnthropic(body);
  if (tools !== null) out.tools = tools;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = anthropicStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = geminiToolChoice(body.tool_choice);
  if (choice !== null) out.toolConfig = choice;
  return out;
}

// ---------------------------------------------------------------------------
// Reading a provider's answer back, in its own wire.
// ---------------------------------------------------------------------------

/** One SSE frame: its event name (null for an unnamed frame) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isObject(v) ? v : null;
  } catch {
    return null;
  }
}

/** An OpenAI JSON completion as an upstream answer. */
function openAIJsonAnswer(body: string): UpstreamAnswer {
  const value = jsonOrNull(body);
  if (value === null) throw new UpstreamFailure(502, 'the provider answered a body that is not JSON');
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isObject(choices[0]) ? choices[0] : {};
  const message = isObject(choice.message) ? choice.message : {};
  const usage = isObject(value.usage) ? value.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    usage: { input: numOrZero(usage.prompt_tokens), output: numOrZero(usage.completion_tokens) },
    stop: openAIFamily(choice.finish_reason),
  };
}

/** An OpenAI SSE stream as an upstream answer. */
function openAIStreamAnswer(text: string): UpstreamAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop = 0;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isObject(chunk.usage)) usage = { input: numOrZero(chunk.usage.prompt_tokens), output: numOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isObject(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isObject(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = openAIFamily(choice.finish_reason);
  }
  return { text: out, usage, stop };
}

/** An Anthropic JSON message as an upstream answer. */
function anthropicJsonAnswer(body: string): UpstreamAnswer {
  const value = jsonOrNull(body);
  if (value === null) throw new UpstreamFailure(502, 'the provider answered a body that is not JSON');
  const content = Array.isArray(value.content) ? value.content : [];
  const text = content
    .flatMap((b) => (isObject(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isObject(value.usage) ? value.usage : {};
  return { text, usage: { input: numOrZero(usage.input_tokens), output: numOrZero(usage.output_tokens) }, stop: anthropicFamily(value.stop_reason) };
}

/** An Anthropic SSE stream as an upstream answer. */
function anthropicStreamAnswer(text: string): UpstreamAnswer {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 0;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isObject(data.message) && isObject(data.message.usage)) input = numOrZero(data.message.usage.input_tokens);
    else if (frame.event === 'content_block_delta' && isObject(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    else if (frame.event === 'message_delta') {
      if (isObject(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicFamily(data.delta.stop_reason);
      if (isObject(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

/** The text of a Gemini candidate's parts. */
function geminiCandidateText(candidate: Json): string {
  const content = isObject(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isObject(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini JSON answer as an upstream answer. */
function geminiJsonAnswer(body: string): UpstreamAnswer {
  const value = jsonOrNull(body);
  if (value === null) throw new UpstreamFailure(502, 'the provider answered a body that is not JSON');
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isObject(candidates[0]) ? candidates[0] : {};
  const usage = isObject(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: geminiCandidateText(candidate),
    usage: { input: numOrZero(usage.promptTokenCount), output: numOrZero(usage.candidatesTokenCount) },
    stop: geminiFamily(candidate.finishReason),
  };
}

/** A Gemini SSE stream as an upstream answer (no [DONE]; the final frame carries usage). */
function geminiStreamAnswer(text: string): UpstreamAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop = 0;
  for (const frame of parseSseFrames(text)) {
    const value = jsonOrNull(frame.data);
    if (value === null) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isObject(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      out += geminiCandidateText(candidate);
      if (typeof candidate.finishReason === 'string') stop = geminiFamily(candidate.finishReason);
    }
    if (isObject(value.usageMetadata)) usage = { input: numOrZero(value.usageMetadata.promptTokenCount), output: numOrZero(value.usageMetadata.candidatesTokenCount) };
  }
  return { text: out, usage, stop };
}

/**
 * One inference call to a provider, in its own wire, returning the parsed
 * answer. Throws UpstreamFailure on any non-2xx status or unreadable answer.
 * A call is sequential: the caller never has two in flight to one provider.
 */
export async function callProvider(envUrl: string, entry: DirectoryEntry, body: Json, stream: boolean, model: string): Promise<UpstreamAnswer> {
  const base = joinBase(envUrl, entry.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(entry.auth, entry.key) };
  let url: string;
  if (entry.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (entry.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(10000) });
  } catch (err) {
    throw new UpstreamFailure(502, `the provider could not be reached: ${(err as Error).message}`);
  }
  const text = await res.text();
  if (!res.ok) throw new UpstreamFailure(res.status, `the provider answered HTTP ${res.status}`);
  if (entry.wire === 'openai') return stream ? openAIStreamAnswer(text) : openAIJsonAnswer(text);
  if (entry.wire === 'anthropic') return stream ? anthropicStreamAnswer(text) : anthropicJsonAnswer(text);
  return stream ? geminiStreamAnswer(text) : geminiJsonAnswer(text);
}

// ---------------------------------------------------------------------------
// Answering the client, in the wire it used.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A provider answer as an OpenAI Chat Completions JSON object. */
export function answerAsOpenAIJson(answer: UpstreamAnswer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: familyToOpenAI(answer.stop), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A provider answer as an OpenAI SSE stream, ending with a usage chunk and [DONE]. */
export function answerAsOpenAIStream(answer: UpstreamAnswer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  // The answer is one short text; one content chunk carries it whole.
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  frames.push(sseEvent(null, chunk({}, familyToOpenAI(answer.stop))));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider answer as an Anthropic Messages JSON object. */
export function answerAsAnthropicJson(answer: UpstreamAnswer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: familyToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** A provider answer as an Anthropic SSE stream, in the canonical event order. */
export function answerAsAnthropicStream(answer: UpstreamAnswer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frames = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  if (answer.text !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: familyToAnthropic(answer.stop), stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** Build the upstream request body for a provider's wire from a client body. */
export function upstreamBody(clientApi: 'openai' | 'anthropic', clientBody: Json, providerWire: Wire, model: string): Json {
  if (providerWire === 'openai') {
    const body = clientApi === 'openai' ? { ...clientBody, model } : anthropicToOpenAI(clientBody, model);
    if (body.stream === true) body.stream_options = { include_usage: true };
    return body;
  }
  if (providerWire === 'anthropic') return clientApi === 'anthropic' ? { ...clientBody, model } : openAIToAnthropic(clientBody, model);
  return clientApi === 'openai' ? openAIToGemini(clientBody, model) : anthropicToGemini(clientBody, model);
}
