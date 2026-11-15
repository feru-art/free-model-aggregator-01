// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the trial's brief, probes each provider's catalog route,
// registers its port and catalog with the env, then serves the client's
// twenty requests, one at a time, each answered by the best free model that
// meets it and still has budget, on that provider's own wire, with the
// answer translated back into the wire the client used.
//
//   GET  /v1/models            the free models, in the wire the auth header names
//   POST /v1/chat/completions  an OpenAI-shaped client, JSON or streamed
//   POST /v1/messages          an Anthropic-shaped client, JSON or streamed
//
// `node main.ts --list` prints the catalog and exits 0 before any server
// starts; outside a trial, with no brief to read, that catalog is empty.
// The server polls GET $MAKEFUN_ENV_URL/work and exits 0 the moment the env
// says the workload is done. A process still running when the wall is spent
// scores zero, so nothing here waits on anything but the env.
//
// Routing lives in lib/router.ts (greedy by rating, evidence-based dead
// rule, conditional same-model retry, fallback-only revival); the wires
// live in lib/wires.ts (faithful translation both ways, usage and stop
// family carried through); the brief and catalog live in lib/brief.ts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  assembleCatalog,
  isObject,
  parseBrief,
  probeCatalog,
  ratingRowFor,
  synthesizedModels,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import { Router, type Candidate } from './lib/router.ts';
import {
  answerAsAnthropicJson,
  answerAsAnthropicStream,
  answerAsOpenAIJson,
  answerAsOpenAIStream,
  anthropicError,
  anthropicModelList,
  callProvider,
  openAIError,
  openAIModelList,
  upstreamBody,
  UpstreamFailure,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wires.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = parseBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog and router. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((entry) => probeCatalog(ENV_URL, entry, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = assembleCatalog(probes, brief.ratings);
  router = new Router(brief, probes);
}

/**
 * The free models across the providers, deduplicated by id. A provider whose
 * probe failed still contributes the models the quality table names for it,
 * exactly as the catalog does: GET /v1/models must hold precisely the set of
 * free ids the catalog registers.
 */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    const listed = probe.ok ? probe.models : synthesizedModels(brief.ratings, probe.entry.id);
    const sorted = [...listed].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      const row = ratingRowFor(brief.ratings, probe.entry.id, model.id);
      const free = row !== null ? row.free : model.free ?? false;
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.entry.id, displayName: model.display_name });
    }
  }
  return out;
}

/** Whether the client's request asks for JSON mode (only an OpenAI body can carry it). */
function asksJsonMode(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isObject(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(clientApi, body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  if (clientApi === 'anthropic') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    let vision = false;
    for (const message of messages) {
      if (!isObject(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isObject(block) && block.type === 'image') vision = true;
      }
    }
    return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: false };
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isObject(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isObject(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isObject(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObject(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, text: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Answer one client request: route it, call providers until one answers,
 * and translate the answer back into the client's wire. Returns false when
 * nothing answered, so the caller can send a 503 in the client's wire.
 */
async function serveRequest(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  if (router === null) return false;
  const stream = body.stream === true;
  const requires = requirementsFor(clientApi, body);
  const jsonMode = asksJsonMode(clientApi, body);
  const candidates = router.candidatesFor(requires, jsonMode, clientApi);
  if (candidates.length === 0) return false;

  let answer: UpstreamAnswer | null = null;
  let answeredBy: Candidate | null = null;
  const attempt = async (candidate: Candidate): Promise<boolean> => {
    const probe = probes.find((p) => p.entry.id === candidate.provider);
    if (probe === undefined) return false;
    try {
      const upstream = upstreamBody(clientApi, body, probe.entry.wire, candidate.model);
      answer = await callProvider(ENV_URL, probe.entry, upstream, stream, candidate.model);
      answeredBy = candidate;
      router?.noteCall(candidate.provider, true);
      return true;
    } catch (err) {
      if (err instanceof UpstreamFailure) {
        router?.noteCall(candidate.provider, false);
        return false;
      }
      throw err;
    }
  };

  const served = await router.serve(candidates, attempt);
  if (served === null || answer === null || answeredBy === null) return false;
  const model = answeredBy.model;
  if (clientApi === 'openai') {
    if (stream) sendText(res, 200, answerAsOpenAIStream(answer, model), 'text/event-stream; charset=utf-8');
    else sendJson(res, 200, answerAsOpenAIJson(answer, model));
  } else {
    if (stream) sendText(res, 200, answerAsAnthropicStream(answer, model), 'text/event-stream; charset=utf-8');
    else sendJson(res, 200, answerAsAnthropicJson(answer, model));
  }
  return true;
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serveRequest('openai', body, res);
  if (!ok) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ok = await serveRequest('anthropic', body, res);
  if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObject(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isObject(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
