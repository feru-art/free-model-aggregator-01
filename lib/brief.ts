// The brief and the catalog: reading the trial's three documents, probing each
// provider's own catalog route, and assembling the one catalog document that
// --list prints and /register carries.
//
// A trial's brief (the file at $MAKEFUN_PROMPT_FILE) is prose wrapped around
// three JSON documents, found by their `schema` member:
//   - the provider directory: ten rows, each with id, wire, auth scheme, the
//     key variable, the key the simulated service takes, and the base path.
//   - the quality table: every model's rating in milli-points, its context
//     length, what it can do (tools, vision, JSON mode) and whether it is free.
//   - the workload manifest: the twenty requests in order, each with its api,
//     whether it streams, its weight and the requirements a model must meet.
//
// The catalog probe speaks each provider's own wire, and a probe that fails
// does not drop its provider from the catalog: the models are synthesized
// from the quality table instead, which is the sealed truth about which
// models are free and how long their context is. Dropping a provider for a
// failed probe would cost its catalog checks outright, where a synthesized
// catalog keeps them. The catalog's base_url is the directory's own path
// (such as /p/groq), never joined to $MAKEFUN_ENV_URL: the env's origin
// differs between a live pass and its replay, and the catalog is part of
// the state export a replay must reproduce byte for byte.
//
// Everything here is a pure function of the brief and the probe answers: no
// clock, no randomness, so --list and the register body build the same
// document.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const DIRECTORY_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isObject = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is its path (or absolute URL). */
export interface DirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What --list prints and what the entry POSTs to /register: one document. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy of it). */
export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsTable {
  rows: RatingRow[];
  best: { id: string; rating: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface ManifestRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: DirectoryEntry[];
  ratings: RatingsTable | null;
  manifest: ManifestRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered. */
export interface Probe {
  entry: DirectoryEntry;
  ok: boolean;
  models: ListedModel[];
  error: string | null;
}

/** What a model can do, from the quality table first and the listing second. */
export interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire never decides the header. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBoolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsWithSchema(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = closingBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isObject(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function directoryEntryOf(v: unknown): DirectoryEntry | null {
  if (!isObject(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a
  // directory spelling an absolute base may use base_path, base_url or base.
  const base =
    asString(v.path) !== '' ? asString(v.path) : asString(v.base_path) !== '' ? asString(v.base_path) : asString(v.base_url) !== '' ? asString(v.base_url) : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, keyEnv: asString(v.key_env), key: asString(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isObject(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestRequestOf(v: unknown): ManifestRequest | null {
  if (!isObject(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isObject(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { minContext: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function parseBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: null, manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsWithSchema(text);
  const directoryDoc = docs.find((d) => d.schema === DIRECTORY_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${DIRECTORY_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = directoryEntryOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsTable | null =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? null
      : {
          rows: ratingsDoc.ratings.flatMap((v) => {
            const r = ratingRowOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isObject(v) || asString(v.id) === '') return [];
                return [{ id: asString(v.id), rating: asNumberOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest: ManifestRequest[] =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = manifestRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingRowFor(ratings: RatingsTable | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.rows.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the
 * sealed truth), the listing's own word where it does not, paid otherwise.
 * A paid model is never routed.
 */
export function isFreeModel(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, else the listing. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/** A listing's price field: true when every price it names is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isObject(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isObject(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context = asNumberOrNull(v.context_length) ?? asNumberOrNull(v.context_window) ?? asNumberOrNull(v.inputTokenLimit);
  return {
    id,
    displayName: asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id,
    free: asBoolOrNull(v.free) ?? zeroPriced(v),
    contextLength: context,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function listedModelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isObject(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeCatalog(envUrl: string, entry: DirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, entry.base);
  const url = entry.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(entry.auth, entry.key) };
  if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { entry, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { entry, ok: true, models: listedModelsIn(entry.wire, body), error: null };
  } catch (err) {
    return { entry, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The models a provider's quality-table rows name, spelled the way a listing
 * would spell them: used when its catalog probe fails, so the provider stays
 * in the catalog (and in GET /v1/models) instead of being dropped for a
 * probe failure. The quality table is the sealed truth about which models
 * are free, what they can do and how long their context is.
 */
export function synthesizedModels(ratings: RatingsTable | null, providerId: string): ListedModel[] {
  if (ratings === null) return [];
  return ratings.rows
    .filter((r) => r.provider === providerId)
    .map((r) => ({ id: r.model, displayName: r.model, free: r.free, contextLength: r.contextLength, tools: r.tools, vision: r.vision, json: r.json }))
    .sort(byModelId);
}

/**
 * The catalog the probes and the quality table make: every provider, with
 * the models its probe listed — or, where the probe failed, the models the
 * quality table names for it — sorted by id and deduplicated.
 */
export function assembleCatalog(probes: readonly Probe[], ratings: RatingsTable | null): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const listed = probe.ok ? probe.models : synthesizedModels(ratings, probe.entry.id);
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...listed].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.entry.id, model.id);
      models.push({ id: model.id, free: isFreeModel(model, row), context_length: row?.contextLength ?? model.contextLength ?? null });
    }
    providers.push({ id: probe.entry.id, base_url: probe.entry.base, key_env: probe.entry.keyEnv, wire: probe.entry.wire, models });
  }
  return { providers };
}
