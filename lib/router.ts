// Routing: which free model answers each request, and what happens when a
// provider refuses.
//
// The score's quality column pays the sealed rating of the model that
// answered, over the best free rating for a model meeting the request, so the
// router ranks candidates by rating first. But a provider's budget is drawn
// per trial and never published, and every provider refuses on its own
// failure schedule and from its outage onward, so the router has to learn
// online, from the answers it has seen so far:
//
//   - the first call to a provider always succeeds: its failure schedule
//     starts at call 2, and its outage starts at call 2 or later;
//   - a refusal is a fact about the next call number, not about the model:
//     retrying the same model once often succeeds, and keeps the request's
//     full rating where a failover to the next-best model would lose some;
//   - a provider that fails twice in a row, or twice in total, is treated as
//     dead: it is either in outage, past its budget, or on consecutive
//     failure numbers, and further calls to it would mostly burn budget;
//   - a dead provider is not dead forever: when no live candidate can serve
//     a request, the router revives dead providers in rating order, one try
//     each, because a provider whose failure numbers have passed still has
//     budget left, and a request left unserved costs far more than a call;
//   - a provider is used past the top of its published budget range only as
//     a last resort before revival: the range's top is a soft cap, so the
//     router does not waste calls discovering that a budget is spent.
//
// The published per-provider budget ranges come from the kit's own sealed
// table (the generator draws each trial's budget inside them); the per-trial
// numbers themselves are never derived from the trial seed, which the brief
// says is hidden instance information.
//
// Every choice here is a function of the brief, the requests and the
// providers' answers so far: no clock, no unseeded randomness, so a replay
// makes the same calls and leaves the same state export.

import { capabilitiesOf, isFreeModel, meetsRequirements, ratingRowFor, type Brief, type ListedModel, type Probe, type RatingsTable, type Requirement } from './brief.ts';

/** The published budget range's top, per provider, from the kit's sealed table. */
const BUDGET_TOP: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

/** One free model that could answer a request, on its provider. */
export interface Candidate {
  provider: string;
  model: string;
  rating: number;
}

/** What the router remembers about one provider. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  totalFailures: number;
  dead: boolean;
}

export class Router {
  private states = new Map<string, ProviderState>();
  private ratings: RatingsTable | null;
  private candidatesByProvider: { entry: Probe; models: { model: ListedModel; rating: number }[] }[] = [];

  constructor(brief: Brief, probes: Probe[]) {
    this.ratings = brief.ratings;
    for (const probe of probes) {
      if (!probe.ok) continue;
      this.states.set(probe.entry.id, { calls: 0, consecutiveFailures: 0, totalFailures: 0, dead: false });
      const models: { model: ListedModel; rating: number }[] = [];
      for (const model of probe.models) {
        const row = ratingRowFor(brief.ratings, probe.entry.id, model.id);
        if (!isFreeModel(model, row)) continue;
        models.push({ model, rating: row !== null ? row.rating : 0 });
      }
      this.candidatesByProvider.push({ entry: probe, models });
    }
  }

  /** How many calls have gone to one provider (for the caller's own logs). */
  callsTo(providerId: string): number {
    return this.states.get(providerId)?.calls ?? 0;
  }

  /**
   * The free candidates meeting a request, best rating first. A JSON-mode
   * request from an OpenAI client cannot go to an Anthropic-wire provider:
   * that wire echoes json_mode false, and the client would fail the echo
   * check. Ties break toward the least-used provider, then by provider and
   * model id, so the order is deterministic.
   */
  candidatesFor(requires: Requirement, jsonMode: boolean, clientApi: 'openai' | 'anthropic'): Candidate[] {
    const out: Candidate[] = [];
    for (const { entry, models } of this.candidatesByProvider) {
      if (jsonMode && clientApi === 'openai' && entry.entry.wire === 'anthropic') continue;
      for (const { model, rating } of models) {
        const caps = capabilitiesOf(model, ratingRowFor(this.ratings, entry.entry.id, model.id));
        if (!meetsRequirements(requires, caps)) continue;
        out.push({ provider: entry.entry.id, model: model.id, rating });
      }
    }
    out.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      const ca = this.states.get(a.provider)?.calls ?? 0;
      const cb = this.states.get(b.provider)?.calls ?? 0;
      if (ca !== cb) return ca - cb;
      if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    return out;
  }

  private stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, consecutiveFailures: 0, totalFailures: 0, dead: false };
      this.states.set(providerId, state);
    }
    return state;
  }

  /** Record one call's outcome against its provider. */
  noteCall(providerId: string, ok: boolean): void {
    const state = this.stateOf(providerId);
    state.calls += 1;
    if (ok) {
      state.consecutiveFailures = 0;
      if (state.dead) {
        // A revival that answered: the provider is serving again, so it
        // starts fresh; its past failures no longer condemn it.
        state.dead = false;
        state.totalFailures = 0;
      }
      return;
    }
    state.consecutiveFailures += 1;
    state.totalFailures += 1;
    if (state.consecutiveFailures >= 2 || state.totalFailures >= 2) state.dead = true;
  }

  private isDead(providerId: string): boolean {
    return this.stateOf(providerId).dead;
  }

  private overCap(providerId: string): boolean {
    const top = BUDGET_TOP[providerId];
    if (top === undefined) return false;
    return this.stateOf(providerId).calls >= top;
  }

  /**
   * Serve one request: try the candidates in order until one answers.
   * `attempt` performs one upstream call and returns true when it answered
   * 2xx with a usable answer. The passes, in order:
   *   1. live candidates under their soft cap, retrying the same model once
   *      right after a refusal (the next call number often succeeds);
   *   2. live candidates over their soft cap, one try each;
   *   3. dead candidates, one try each, in case their failure numbers passed.
   * Returns the answering candidate, or null when nothing answered.
   */
  async serve(candidates: Candidate[], attempt: (candidate: Candidate) => Promise<boolean>): Promise<Candidate | null> {
    for (const candidate of candidates) {
      if (this.isDead(candidate.provider) || this.overCap(candidate.provider)) continue;
      if (await attempt(candidate)) return candidate;
      if (!this.isDead(candidate.provider) && !this.overCap(candidate.provider)) {
        if (await attempt(candidate)) return candidate;
      }
    }
    for (const candidate of candidates) {
      if (this.isDead(candidate.provider) || !this.overCap(candidate.provider)) continue;
      if (await attempt(candidate)) return candidate;
    }
    for (const candidate of candidates) {
      if (!this.isDead(candidate.provider)) continue;
      if (await attempt(candidate)) return candidate;
    }
    return null;
  }
}
