// gta6-trophy-rarity-forecast-01 v3 — cohort-convolution trophy forecaster.
//
// The program reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and
// writes <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output). Each case
// holds the house's reads of the Exophase GTA VI trophy page before a
// checkpoint; the forecast is five quantiles of every trophy's earn rate
// (percent of tracked players who earned it) at the case's target.
//
// Model, in one paragraph: Exophase refreshes in steps, so the reads are
// first collapsed into distinct refresh states. Three candidate forecasts
// are built per trophy, all in logit space (the scoring scale): (A) a
// per-trophy regression of logit(rate) on log(players) with the slope
// shrunk toward its tier mean, extrapolated to a saturating player
// forecast; (B) a damped local velocity from the freshest refresh-to-
// refresh logit moves; (C) a cohort convolution — players arrive as
// cohorts sized by the increments of players_tracked, each cohort earns
// trophy i following a Weibull CDF in hours since joining with a shared
// shape kappa and a per-trophy scale lambda regularized toward its tier,
// and the forecast convolves the player-growth curve with those CDFs.
// A rolling-origin backtest inside the case's own history blends the
// candidates, shifts a global bias, and scales the spread. Histories with
// fewer than two distinct pages fall back to a dilution-aware wide band
// around the last rates. Node 22, standard library only, deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

// ---------- small numeric helpers ----------

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

/** Percent clamped to the scorer's logit-safe range. */
const clampPct = (pct: number): number => clamp(pct, 0.01, 99.99);

const logitPct = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};

const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -16, 16)));

const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/**
 * Quantization-aware spread floor. Rates print to 2 decimals, so one
 * print step of 0.01 points at a rate of r percent is about 0.01 / r in
 * logit units: at 0.05% a single step is ~0.2 logit, at 0.5% only ~0.02.
 * A forecast for a rare trophy therefore carries an irreducible logit
 * spread from the printing grid alone, however calm its history looks.
 * Returns that floor (in logit units) for a median rate in percent;
 * it is negligible above ~0.5% and capped for the very rarest rows.
 */
function quantSpreadFloor(ratePct: number): number {
  const r = clamp(ratePct, 0.01, 100);
  if (r >= 0.5) return 0.014 / r; // <= 0.028, a whisper above 0.5%
  return clamp(0.014 / Math.max(r, 0.02), 0, 0.9);
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const a = [...xs].sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 === 1 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Inverse standard normal CDF (Acklam's rational approximation). */
function zFromTau(tau: number): number {
  if (tau <= 0) return -4;
  if (tau >= 1) return 4;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  if (tau < plow) {
    const q = Math.sqrt(-2 * Math.log(tau));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (tau > 1 - plow) {
    const q = Math.sqrt(-2 * Math.log(1 - tau));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = tau - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---------- case shape ----------

interface TrophyInfo { id: number; tier: string }
interface ReadRec { slot?: string; at?: string; players_tracked: number; rates: Array<number | null> }
interface CaseT {
  id: string;
  horizon_minutes: number;
  levels?: number[];
  game?: { release_date?: string };
  trophies: TrophyInfo[];
  history: ReadRec[];
}

interface State { t: number; players: number; rates: number[] }

function parseTime(s: string | undefined): number {
  if (!s) return 0;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : 0;
}

/**
 * Collapse the raw reads into distinct refresh states: repeated reads of
 * the same page (same players_tracked and same rates) become one state,
 * stamped with the first time that page was seen. Null rates are filled
 * forward, then backward, so every state has a full rate row.
 */
function buildStates(c: CaseT): State[] {
  const n = c.trophies.length;
  const lastKnown = new Array<number>(n).fill(NaN);
  const filled: number[][] = [];
  for (const h of c.history) {
    const row = new Array<number>(n).fill(NaN);
    for (let i = 0; i < n; i++) {
      const v = h.rates[i];
      if (v !== null && v !== undefined && Number.isFinite(v)) lastKnown[i] = v;
      row[i] = lastKnown[i];
    }
    filled.push(row);
  }
  const nextKnown = new Array<number>(n).fill(0);
  for (let k = filled.length - 1; k >= 0; k--) {
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(filled[k][i])) nextKnown[i] = filled[k][i];
      else filled[k][i] = nextKnown[i];
    }
  }
  const seen = new Set<string>();
  const states: State[] = [];
  c.history.forEach((h, idx) => {
    const key = `${h.players_tracked}|${filled[idx].join(',')}`;
    if (seen.has(key)) return;
    seen.add(key);
    states.push({ t: parseTime(h.at ?? h.slot), players: h.players_tracked, rates: filled[idx] });
  });
  states.sort((a, b) => a.t - b.t);
  return states;
}

/** Keep at most `cap` states, evenly spaced, always including first and last. */
function thinStates(states: State[], cap: number): State[] {
  if (states.length <= cap) return states;
  const out: State[] = [];
  const step = (states.length - 1) / (cap - 1);
  for (let k = 0; k < cap; k++) out.push(states[Math.round(k * step)]);
  return out;
}

// ---------- player-count forecast ----------

interface PlayerFc {
  target: number;
  /** Forecast players `u` hours after the last state (0..horizon). */
  at: (u: number) => number;
}

function forecastPlayers(states: State[], horizonH: number): PlayerFc {
  const last = states[states.length - 1];
  const pLast = Math.max(1, last.players);
  if (states.length < 2 || horizonH <= 0) {
    return { target: pLast, at: () => pLast };
  }
  const xs = states.map((s) => (s.t - last.t) / 3600000);
  const ys = states.map((s) => Math.log(Math.max(1, s.players)));

  // Damped log-linear trend.
  const mx = mean(xs), my = mean(ys);
  let sxx = 0, sxy = 0;
  for (let k = 0; k < xs.length; k++) { sxx += (xs[k] - mx) ** 2; sxy += (xs[k] - mx) * (ys[k] - my); }
  const slope = sxx > 1e-9 ? sxy / sxx : 0;
  const tauDamp = 30;
  const logTrend = Math.log(pLast) + slope * tauDamp * (1 - Math.exp(-horizonH / tauDamp));

  // Saturating growth: grid over the ceiling K, linearize log(K - P).
  let bestSat = NaN, bestSse = Infinity;
  const pMax = Math.max(...states.map((s) => s.players));
  for (const mult of [1.25, 1.6, 2.2, 3.2, 5, 8, 13, 22, 40, 80]) {
    const K = Math.max(pLast * mult, pMax * 1.04 + 10);
    const ly = states.map((s) => Math.log(Math.max(1, K - s.players)));
    const mly = mean(ly);
    let a2 = 0, b2 = 0;
    for (let k = 0; k < xs.length; k++) { a2 += (xs[k] - mx) ** 2; b2 += (xs[k] - mx) * (ly[k] - mly); }
    if (a2 <= 1e-9) continue;
    const b = b2 / a2;
    if (b >= 0) continue; // gap to the ceiling must shrink
    const a = mly - b * mx;
    let sse = 0;
    for (let k = 0; k < xs.length; k++) sse += (ly[k] - (a + b * xs[k])) ** 2;
    if (sse < bestSse) {
      bestSse = sse;
      bestSat = K - Math.exp(a + b * horizonH);
    }
  }
  let logTarget = logTrend;
  if (Number.isFinite(bestSat) && bestSat > pLast * 0.85) {
    logTarget = 0.5 * logTrend + 0.5 * Math.log(bestSat);
  }
  const target = clamp(Math.exp(logTarget), pLast * 0.97, pLast * 80);
  const tauC = clamp(horizonH / 2.2, 4, 60);
  const denom = 1 - Math.exp(-horizonH / tauC);
  const logRatio = Math.log(target / pLast);
  return {
    target,
    at: (u: number) => {
      if (horizonH <= 0) return pLast;
      const frac = denom > 1e-9 ? (1 - Math.exp(-clamp(u, 0, horizonH) / tauC)) / denom : u / horizonH;
      return pLast * Math.exp(logRatio * frac);
    },
  };
}

// ---------- candidate A: logit(rate) on log(players), tier-shrunk slope ----------

function candidateA(states: State[], tiers: string[], logPTarget: number): number[] {
  const n = tiers.length;
  const S = states.length;
  const last = states[S - 1];
  const xs = states.map((s) => Math.log(Math.max(1, s.players)));
  const mx = mean(xs);
  let sxx = 0;
  for (const x of xs) sxx += (x - mx) ** 2;
  const slopes = new Array<number>(n).fill(0);
  if (sxx > 1e-9) {
    for (let i = 0; i < n; i++) {
      const ys = states.map((s) => logitPct(s.rates[i]));
      const my = mean(ys);
      let sxy = 0;
      for (let k = 0; k < S; k++) sxy += (xs[k] - mx) * (ys[k] - my);
      slopes[i] = clamp(sxy / sxx, -4, 4);
    }
  }
  const tierMean = new Map<string, number>();
  const byTier = new Map<string, number[]>();
  tiers.forEach((t, i) => {
    const arr = byTier.get(t) ?? [];
    arr.push(slopes[i]);
    byTier.set(t, arr);
  });
  const globalMean = mean(slopes);
  for (const [t, arr] of byTier) tierMean.set(t, mean(arr));
  const dLogP = logPTarget - Math.log(Math.max(1, last.players));
  const wOwn = Math.max(0, S - 2);
  return tiers.map((t, i) => {
    const shrunk = (wOwn * slopes[i] + 2 * (tierMean.get(t) ?? globalMean) + globalMean) / (wOwn + 3);
    return clamp(logitPct(last.rates[i]) + shrunk * dLogP, -11.5, 4.65);
  });
}

// ---------- candidate B: damped local velocity ----------

function candidateB(states: State[], tiers: string[], horizonH: number): number[] {
  const n = tiers.length;
  const S = states.length;
  const last = states[S - 1];
  if (S < 2) return tiers.map((_, i) => logitPct(last.rates[i]));
  // Per-transition logit velocities (per hour), most recent last.
  const trans: Array<{ dt: number; dv: number[] }> = [];
  for (let k = 1; k < S; k++) {
    const dt = (states[k].t - states[k - 1].t) / 3600000;
    if (dt <= 0.01) continue;
    trans.push({
      dt,
      dv: tiers.map((_, i) => (logitPct(states[k].rates[i]) - logitPct(states[k - 1].rates[i])) / dt),
    });
  }
  if (trans.length === 0) return tiers.map((_, i) => logitPct(last.rates[i]));
  const recent = trans.slice(-5);
  const vel = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let wsum = 0, vsum = 0;
    recent.forEach((tr, idx) => {
      const w = Math.pow(0.62, recent.length - 1 - idx);
      wsum += w;
      vsum += w * tr.dv[i];
    });
    vel[i] = clamp(wsum > 0 ? vsum / wsum : 0, -0.35, 0.35);
  }
  const tierVel = new Map<string, number>();
  const byTier = new Map<string, number[]>();
  tiers.forEach((t, i) => {
    const arr = byTier.get(t) ?? [];
    arr.push(vel[i]);
    byTier.set(t, arr);
  });
  const globalVel = mean(vel);
  for (const [t, arr] of byTier) tierVel.set(t, mean(arr));
  const tauD = 26;
  const gain = tauD * (1 - Math.exp(-horizonH / tauD));
  return tiers.map((t, i) => {
    const shrunk = (2 * vel[i] + (tierVel.get(t) ?? globalVel) + globalVel) / 4;
    return clamp(logitPct(last.rates[i]) + shrunk * gain, -11.5, 4.65);
  });
}

// ---------- candidate C: cohort convolution with Weibull earning curves ----------

interface CohortFit {
  kappa: number;
  lambda: number[]; // per trophy, hours, tier-regularized
  states: State[]; // thinned states the fit used
  joins: number[]; // cohort join times, hours since first thinned state
  sizes: number[]; // cohort sizes
}

const weibullF = (ageH: number, lambda: number, kappa: number): number => {
  if (ageH <= 0) return 0;
  const x = ageH / lambda;
  const v = 1 - Math.exp(-Math.pow(x, kappa));
  return clamp(v, 0, 1);
};

function fitCohort(statesIn: State[], tiers: string[], releaseMs: number, coarse: boolean): CohortFit {
  const states = thinStates(statesIn, coarse ? 24 : 40);
  const S = states.length;
  const n = tiers.length;
  const t0 = states[0].t;
  const tH = states.map((s) => (s.t - t0) / 3600000);
  // Cohort join times: the initial cohort is dated to the release (or a
  // 10-hour head start when the release stamp is unusable); later cohorts
  // join when their players first appear in a refresh state.
  let j0 = -10;
  if (Number.isFinite(releaseMs) && releaseMs > 0) {
    const rel = (releaseMs - t0) / 3600000;
    if (rel >= -120 && rel <= 2) j0 = Math.min(rel, 0);
  }
  const joins: number[] = [j0];
  const sizes: number[] = [Math.max(0, states[0].players)];
  for (let k = 1; k < S; k++) {
    joins.push(tH[k]);
    sizes.push(Math.max(0, states[k].players - states[k - 1].players));
  }
  // Age of cohort c at state s.
  const ages: number[][] = joins.map((j) => tH.map((t) => Math.max(0, t - j)));
  // Observed earners per trophy per state.
  const obs: number[][] = tiers.map((_, i) => states.map((s) => (s.rates[i] / 100) * s.players));

  const kappas = coarse ? [0.7, 1.0, 1.4] : [0.55, 0.75, 1.0, 1.35, 1.8];
  const nLam = coarse ? 14 : 24;
  const lamGrid: number[] = [];
  for (let k = 0; k < nLam; k++) lamGrid.push(Math.exp(Math.log(1.5) + (Math.log(3500) - Math.log(1.5)) * (k / (nLam - 1))));

  // sse[trophy][kappaIdx] and best lambda per (trophy, kappa).
  const bestLam: number[][] = tiers.map(() => new Array<number>(kappas.length).fill(200));
  const bestSse: number[][] = tiers.map(() => new Array<number>(kappas.length).fill(Infinity));

  const sseFor = (i: number, lambda: number, kappa: number): number => {
    let sse = 0;
    for (let s = 0; s < S; s++) {
      let pred = 0;
      for (let cc = 0; cc < S; cc++) {
        if (sizes[cc] === 0) continue;
        pred += sizes[cc] * weibullF(ages[cc][s], lambda, kappa);
      }
      const frac = (pred - obs[i][s]) / Math.max(1, states[s].players);
      sse += frac * frac;
    }
    return sse / S;
  };

  kappas.forEach((kappa, ki) => {
    for (const lambda of lamGrid) {
      // Precompute the CDF matrix once per (kappa, lambda).
      const F: number[][] = ages.map((row) => row.map((a) => weibullF(a, lambda, kappa)));
      for (let i = 0; i < n; i++) {
        let sse = 0;
        for (let s = 0; s < S; s++) {
          let pred = 0;
          for (let cc = 0; cc < S; cc++) if (sizes[cc] !== 0) pred += sizes[cc] * F[cc][s];
          const frac = (pred - obs[i][s]) / Math.max(1, states[s].players);
          sse += frac * frac;
        }
        sse /= S;
        if (sse < bestSse[i][ki]) { bestSse[i][ki] = sse; bestLam[i][ki] = lambda; }
      }
    }
  });

  // Shared kappa: the one with the lowest total error across trophies.
  let kappaIdx = 0, kappaScore = Infinity;
  kappas.forEach((_, ki) => {
    let tot = 0;
    for (let i = 0; i < n; i++) tot += bestSse[i][ki];
    if (tot < kappaScore) { kappaScore = tot; kappaIdx = ki; }
  });
  const kappa = kappas[kappaIdx];
  const lambda = bestLam.map((row) => row[kappaIdx]);

  // Local refinement of each lambda at the shared kappa (full fits only).
  if (!coarse) {
    for (let i = 0; i < n; i++) {
      for (let round = 0; round < 2; round++) {
        let improved = false;
        for (const f of [0.74, 0.86, 1.16, 1.35]) {
          const cand = clamp(lambda[i] * f, 0.8, 6000);
          const s = sseFor(i, cand, kappa);
          if (s < sseFor(i, lambda[i], kappa)) { lambda[i] = cand; improved = true; }
        }
        if (!improved) break;
      }
    }
  }

  // Regularize each lambda toward its tier's median lambda (log space).
  const byTier = new Map<string, number[]>();
  tiers.forEach((t, i) => {
    const arr = byTier.get(t) ?? [];
    arr.push(lambda[i]);
    byTier.set(t, arr);
  });
  const globalMed = median(lambda);
  const tierMed = new Map<string, number>();
  for (const [t, arr] of byTier) tierMed.set(t, arr.length >= 3 ? median(arr) : globalMed);
  const wFit = clamp((S - 1) / 2, 0.5, 4);
  for (let i = 0; i < n; i++) {
    const tm = tierMed.get(tiers[i]) ?? globalMed;
    lambda[i] = Math.exp((wFit * Math.log(lambda[i]) + 2.5 * Math.log(tm)) / (wFit + 2.5));
  }
  return { kappa, lambda, states, joins, sizes };
}

function cohortForecast(fit: CohortFit, pfc: PlayerFc, horizonH: number): number[] {
  const { kappa, lambda, states, joins, sizes } = fit;
  const S = states.length;
  const tLastH = (states[S - 1].t - states[0].t) / 3600000;
  // Future cohorts: slice the horizon and size each slice from the
  // player-growth curve, joining at the slice midpoint.
  const M = 12;
  const futJoins: number[] = [];
  const futSizes: number[] = [];
  for (let m = 0; m < M; m++) {
    const u0 = (horizonH * m) / M, u1 = (horizonH * (m + 1)) / M;
    futJoins.push(tLastH + (u0 + u1) / 2);
    futSizes.push(Math.max(0, pfc.at(u1) - pfc.at(u0)));
  }
  const targetH = tLastH + horizonH;
  return lambda.map((lam) => {
    let earners = 0;
    for (let cc = 0; cc < S; cc++) {
      if (sizes[cc] === 0) continue;
      earners += sizes[cc] * weibullF(targetH - joins[cc], lam, kappa);
    }
    for (let m = 0; m < M; m++) {
      if (futSizes[m] === 0) continue;
      earners += futSizes[m] * weibullF(targetH - futJoins[m], lam, kappa);
    }
    return clamp((100 * earners) / Math.max(1, pfc.target), 0, 100);
  });
}

// ---------- pipeline + rolling-origin backtest ----------

interface PipelineOut {
  a: number[];
  b: number[];
  c: number[];
  cValid: boolean;
  pTarget: number;
}

function runPipeline(states: State[], tiers: string[], horizonH: number, releaseMs: number, coarse: boolean): PipelineOut {
  const pfc = forecastPlayers(states, horizonH);
  const a = candidateA(states, tiers, Math.log(pfc.target));
  const b = candidateB(states, tiers, horizonH);
  const distinctPlayers = new Set(states.map((s) => s.players)).size;
  let c = a.slice();
  let cValid = false;
  if (states.length >= 3 && distinctPlayers >= 2) {
    try {
      const fit = fitCohort(states, tiers, releaseMs, coarse);
      const rates = cohortForecast(fit, pfc, horizonH);
      c = rates.map((r) => clamp(logitPct(r), -11.5, 4.65));
      cValid = true;
    } catch {
      cValid = false;
    }
  }
  return { a, b, c, cValid, pTarget: pfc.target };
}

interface Backtest {
  wA: number; wB: number; wC: number;
  bias: number;
  sigmaGlobal: number;
  sigmaTier: Map<string, number>;
  horizonRef: number;
  ok: boolean;
}

function backtest(states: State[], tiers: string[], releaseMs: number): Backtest {
  const fallback: Backtest = {
    wA: 0.38, wB: 0.24, wC: 0.38, bias: 0, sigmaGlobal: 0.6,
    sigmaTier: new Map(), horizonRef: 0, ok: false,
  };
  const S = states.length;
  if (S < 6) return fallback;
  const cutSet = new Set<number>();
  for (const f of [0.4, 0.6, 0.8]) {
    const idx = Math.floor(S * f);
    if (idx >= 3 && idx <= S - 2) cutSet.add(idx);
  }
  if (cutSet.size === 0) return fallback;
  const last = states[S - 1];
  const actual = tiers.map((_, i) => logitPct(last.rates[i]));
  const errA: number[] = [], errB: number[] = [], errC: number[] = [];
  const resid: number[] = [];
  const residTier = new Map<string, number[]>();
  const horizons: number[] = [];
  let cUsable = true;
  for (const cut of [...cutSet].sort((x, y) => x - y)) {
    const sub = states.slice(0, cut + 1);
    const hBt = (last.t - states[cut].t) / 3600000;
    if (hBt <= 0.05) continue;
    horizons.push(hBt);
    const out = runPipeline(sub, tiers, hBt, releaseMs, true);
    if (!out.cValid) cUsable = false;
    for (let i = 0; i < tiers.length; i++) {
      errA.push(Math.abs(out.a[i] - actual[i]));
      errB.push(Math.abs(out.b[i] - actual[i]));
      if (out.cValid) errC.push(Math.abs(out.c[i] - actual[i]));
      const ens = out.cValid ? (out.a[i] + out.b[i] + out.c[i]) / 3 : (out.a[i] + out.b[i]) / 2;
      const r = actual[i] - ens;
      resid.push(r);
      const arr = residTier.get(tiers[i]) ?? [];
      arr.push(r);
      residTier.set(tiers[i], arr);
    }
  }
  if (resid.length === 0) return fallback;
  const maeA = mean(errA), maeB = mean(errB), maeC = errC.length ? mean(errC) : Infinity;
  let wA = Math.exp(-maeA / 0.3), wB = Math.exp(-maeB / 0.3);
  let wC = cUsable && Number.isFinite(maeC) ? Math.exp(-maeC / 0.3) : 0;
  if (wC > 0) { wA = Math.max(wA, 0.07); wB = Math.max(wB, 0.07); wC = Math.max(wC, 0.07); }
  else { wA = Math.max(wA, 0.07); wB = Math.max(wB, 0.07); }
  const wSum = wA + wB + wC;
  wA /= wSum; wB /= wSum; wC /= wSum;
  const med = median(resid);
  const bias = clamp(med * (resid.length / (resid.length + 8)), -0.5, 0.5);
  const mad = median(resid.map((r) => Math.abs(r - med)));
  const sigmaGlobal = clamp(1.4826 * mad, 0.25, 2.4);
  const sigmaTier = new Map<string, number>();
  for (const [t, arr] of residTier) {
    if (arr.length >= 8) {
      const m2 = median(arr);
      sigmaTier.set(t, clamp(1.4826 * median(arr.map((r) => Math.abs(r - m2))), 0.25, 2.4));
    }
  }
  return { wA, wB, wC, bias, sigmaGlobal, sigmaTier, horizonRef: median(horizons), ok: true };
}

// ---------- per-case forecast ----------

function forecastCase(c: CaseT): Array<{ id: number; q: number[] }> {
  const tiers = c.trophies.map((t) => t.tier);
  const levels = c.levels && c.levels.length === 5 ? c.levels : LEVELS;
  const zs = levels.map((tau) => zFromTau(tau));
  const horizonH = c.horizon_minutes / 60;
  const releaseMs = c.game?.release_date ? parseTime(c.game.release_date) : NaN;
  const states = buildStates(c);
  const lastRead = c.history[c.history.length - 1];
  const lastRates = c.trophies.map((_, i) => {
    const v = lastRead?.rates[i];
    return v !== null && v !== undefined && Number.isFinite(v) ? v : (states[states.length - 1]?.rates[i] ?? 0);
  });

  const emit = (medianLogit: number[], sigLow: number[], sigHigh: number[]) =>
    c.trophies.map((t, i) => {
      const q = zs.map((z) => {
        const sig = z < 0 ? sigLow[i] : sigHigh[i];
        return round4(clamp(invLogitPct(medianLogit[i] + z * sig), 0, 100));
      });
      for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
      return { id: t.id, q };
    });

  const distinctPlayers = new Set(states.map((s) => s.players)).size;

  // Degenerate history: fewer than two distinct pages. No slope, no
  // velocity, no cohort structure can be identified — fall back to the
  // last rates with a dilution-aware asymmetric band: an unseen refresh
  // can multiply the tracked players several-fold (diluting every rate
  // whose earners lag) while progression pushes the other way.
  if (states.length < 2 || distinctPlayers < 2) {
    const hScale = Math.sqrt(clamp(horizonH, 0.25, 36) / 1.5);
    const sigLowBase = 1.02 + 0.22 * Math.sqrt(horizonH / 24);
    const sigHighBase = 0.55 + 0.16 * Math.sqrt(horizonH / 24);
    const medianLogit: number[] = [], sigLow: number[] = [], sigHigh: number[] = [];
    for (let i = 0; i < tiers.length; i++) {
      const r = lastRates[i];
      if (r >= 99.99) {
        medianLogit.push(logitPct(99.99)); sigLow.push(0.12); sigHigh.push(0.12);
        continue;
      }
      if (r <= 0) {
        medianLogit.push(logitPct(0.05)); sigLow.push(0.7); sigHigh.push(1.15);
        continue;
      }
      const rarity = Math.exp(-r / 3); // 1 for very rare, -> 0 for common
      const shift = -(0.34 + 0.62 * rarity) * hScale;
      medianLogit.push(clamp(logitPct(r) + shift, -11.5, 4.65));
      const qFloor = quantSpreadFloor(r);
      sigLow.push(Math.sqrt((sigLowBase + 0.35 * rarity) ** 2 + qFloor * qFloor));
      sigHigh.push(Math.sqrt((sigHighBase + 0.2 * (1 - rarity)) ** 2 + qFloor * qFloor));
    }
    return emit(medianLogit, sigLow, sigHigh);
  }

  // Rich history: ensemble of A, B and C, calibrated by the backtest.
  const out = runPipeline(states, tiers, horizonH, releaseMs, false);
  const bt = backtest(states, tiers, releaseMs);
  let { wA, wB, wC } = bt;
  if (!out.cValid) { const s = wA + wB; wA = s > 0 ? wA / s : 0.6; wB = s > 0 ? wB / s : 0.4; wC = 0; }

  // Per-trophy step dispersion: how jumpy each trophy's logit rate is
  // between refreshes, and the median refresh gap (for step risk).
  const gaps: number[] = [];
  for (let k = 1; k < states.length; k++) {
    const g = (states[k].t - states[k - 1].t) / 3600000;
    if (g > 0) gaps.push(g);
  }
  const medianGap = gaps.length ? median(gaps) : 6;
  const stepAbs = tiers.map((_, i) => {
    const jumps: number[] = [];
    for (let k = 1; k < states.length; k++) {
      jumps.push(Math.abs(logitPct(states[k].rates[i]) - logitPct(states[k - 1].rates[i])));
    }
    return jumps.length ? mean(jumps) : 0;
  });
  const tierStep = new Map<string, number>();
  {
    const byTier = new Map<string, number[]>();
    tiers.forEach((t, i) => {
      const arr = byTier.get(t) ?? [];
      arr.push(stepAbs[i]);
      byTier.set(t, arr);
    });
    for (const [t, arr] of byTier) tierStep.set(t, mean(arr));
  }
  const expRefresh = clamp(horizonH / Math.max(medianGap, 1), 0, 14);
  const horizonScale = bt.ok && bt.horizonRef > 0
    ? Math.sqrt(clamp(horizonH / bt.horizonRef, 0.5, 6))
    : 1;

  const medianLogit: number[] = [], sigmas: number[] = [];
  for (let i = 0; i < tiers.length; i++) {
    const m = wA * out.a[i] + wB * out.b[i] + wC * out.c[i] + bt.bias;
    medianLogit.push(clamp(m, -11.5, 4.65));
    const sigTier = bt.sigmaTier.get(tiers[i]) ?? bt.sigmaGlobal;
    const base = Math.sqrt(0.65 * bt.sigmaGlobal ** 2 + 0.35 * sigTier ** 2) * horizonScale;
    const stepTerm = (0.55 * stepAbs[i] + 0.45 * (tierStep.get(tiers[i]) ?? 0)) * Math.sqrt(expRefresh) * 0.5;
    const qFloor = quantSpreadFloor(invLogitPct(medianLogit[i]));
    sigmas.push(clamp(Math.sqrt(base * base + stepTerm * stepTerm + qFloor * qFloor + 0.04), 0.2, 2.8));
  }
  return emit(medianLogit, sigmas, sigmas);
}

/** Last-resort valid forecast: persistence with a wide random-walk band. */
function fallbackCase(c: CaseT): Array<{ id: number; q: number[] }> {
  const last = c.history[c.history.length - 1];
  const spread = 0.5 * Math.sqrt(c.horizon_minutes / 1440) + 0.35;
  const zs = LEVELS.map((tau) => zFromTau(tau));
  return c.trophies.map((t, i) => {
    const r = last?.rates[i] ?? 0;
    const l = logitPct(typeof r === 'number' && Number.isFinite(r) ? r : 0);
    const q = zs.map((z) => round4(clamp(invLogitPct(l + z * spread), 0, 100)));
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    return { id: t.id, q };
  });
}

// ---------- main ----------

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseT;
  let trophies: Array<{ id: number; q: number[] }>;
  try {
    trophies = forecastCase(c);
  } catch {
    trophies = fallbackCase(c);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
