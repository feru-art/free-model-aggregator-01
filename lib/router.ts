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
//     full rating where a failover to the next-best model would lose some —
//     but only when the provider is not proven dead and the refusal came
//     below the bottom of its published budget range, where it cannot be a
//     spent budget; past that bottom, the next call number is likely spent
//     too, and a failover is the cheaper move;
//   - a provider is dead only on proof: a refusal on a call at or past the
//     top of its published budget range (its budget cannot still be
//     unspent), or a third refusal in total (its failure schedule holds at
//     most two numbers, so a third proves an outage or a spent budget).
//     Two refusals in a row, or two in total, prove nothing: both can be
//     scheduled failures, and a provider dropped on them still has budget
//     left to serve the requests only it can meet;
//   - a dead provider is revived only as a fallback, when no live candidate
//     served the request at all: there is no separate revival pass, because
//     a provider written off on proof rarely answers again, and a request
//     left unserved costs far more than a call;
//   - a provider is used past the top of its published budget range only as
//     a last resort in that same fallback: the range's top is a soft cap,
//     so the router does not waste calls discovering that a budget is
//     spent;
//   - ties between candidates break toward the most budget left (the top
//     of the range minus the calls made), then by provider and model id,
//     so the order is deterministic and the deepest providers carry the
//     requests first.
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

/** The published budget range's bottom, per provider, from the same table. */
const BUDGET_BOTTOM: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
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
      this.states.set(probe.entry.id, { calls: 0, totalFailures: 0, dead: false });
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
   * check. Ties break toward the most budget left (the top of the range
   * minus the calls made), then by provider and model id, so the order is
   * deterministic.
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
      const remaining = (provider: string): number => (BUDGET_TOP[provider] ?? 0) - (this.states.get(provider)?.calls ?? 0);
      const ra = remaining(a.provider);
      const rb = remaining(b.provider);
      if (rb !== ra) return rb - ra;
      if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    return out;
  }

  private stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, totalFailures: 0, dead: false };
      this.states.set(providerId, state);
    }
    return state;
  }

  /**
   * Record one call's outcome against its provider. A provider is declared
   * dead only on proof: a refusal on a call at or past the top of its
   * published budget range (its budget cannot still be unspent), or a third
   * refusal in total (its failure schedule holds at most two numbers, so a
   * third proves an outage or a spent budget). Fewer failures prove
   * nothing: both can be scheduled failures.
   */
  noteCall(providerId: string, ok: boolean): void {
    const state = this.stateOf(providerId);
    state.calls += 1;
    if (ok) {
      if (state.dead) {
        // A revival that answered: the provider is serving again, so it
        // starts fresh; its past failures no longer condemn it.
        state.dead = false;
        state.totalFailures = 0;
      }
      return;
    }
    state.totalFailures += 1;
    const top = BUDGET_TOP[providerId];
    if (state.totalFailures >= 3 || (top !== undefined && state.calls >= top)) state.dead = true;
  }

  private isDead(providerId: string): boolean {
    return this.stateOf(providerId).dead;
  }

  private overCap(providerId: string): boolean {
    const top = BUDGET_TOP[providerId];
    if (top === undefined) return false;
    return this.stateOf(providerId).calls >= top;
  }

  /** Whether a provider's calls are still below the bottom of its budget range. */
  private belowMin(providerId: string): boolean {
    const bottom = BUDGET_BOTTOM[providerId];
    if (bottom === undefined) return false;
    return this.stateOf(providerId).calls < bottom;
  }

  /**
   * Serve one request: try the candidates in order until one answers.
   * `attempt` performs one upstream call and returns true when it answered
   * 2xx with a usable answer. The passes, in order:
   *   1. live candidates under their soft cap, retrying the same model once
   *      right after a refusal — but only when the provider is not proven
   *      dead and the refusal came below the bottom of its budget range,
   *      where it cannot be a spent budget;
   *   2. a single fallback, only when pass 1 failed totally: every remaining
   *      candidate (over-cap live and dead alike) in rating order, one try
   *      each. There is no separate revival pass: a provider written off on
   *      proof is tried only here, where the alternative is an unserved
   *      request.
   * Returns the answering candidate, or null when nothing answered.
   */
  async serve(candidates: Candidate[], attempt: (candidate: Candidate) => Promise<boolean>): Promise<Candidate | null> {
    for (const candidate of candidates) {
      if (this.isDead(candidate.provider) || this.overCap(candidate.provider)) continue;
      if (await attempt(candidate)) return candidate;
      if (!this.isDead(candidate.provider) && this.belowMin(candidate.provider)) {
        if (await attempt(candidate)) return candidate;
      }
    }
    for (const candidate of candidates) {
      if (!this.isDead(candidate.provider) && !this.overCap(candidate.provider)) continue;
      if (await attempt(candidate)) return candidate;
    }
    return null;
  }
}
