# gta6-trophy-rarity-forecast-01-v3 — cohort-convolution trophy forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acilpiw/gta6-trophy-rarity-forecast-01-v3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acilpiw/gta6-trophy-rarity-forecast-01-v3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**:
forecast the Exophase earn rate of every GTA VI PlayStation 5 trophy at
2026-11-26T00:00:00Z. Competition page:
https://make.fun/c/gta6-trophy-rarity-forecast-01

This is the third-generation model. Generation 2 collapsed the reads
into refresh states, extrapolated per-trophy logit slopes against player
growth, and calibrated itself with an in-history backtest — a design
that scored a 0.363 practice loss against the house baseline's ~1.51.
Generation 3 keeps that skeleton and adds the mechanism the first two
generations only approximated: players do not arrive all at once, and a
trophy's earn rate is what a stream of arriving cohorts has had time to
earn. The live page on 2026-11-20 shows why this matters — early-game
trophies (Chapter 1 at 7.99%, the fishing challenge at 10.57%) climbed
through the player explosion while most late-game trophies were diluted
to 0.01–0.02%.

## How it works

Everything happens in logit space, the scale the score is charged on.

1. **Refresh states.** Repeated reads of the same page are collapsed, so
   the model sees the staircase Exophase actually published: distinct
   pages with their player counts and timestamps.
2. **Player forecast.** The tracked-player count at the target comes
   from a saturating growth fit (a grid over the ceiling, scored by fit
   quality) blended with a damped log-linear trend, shaped as a
   front-loaded curve over the horizon so future cohorts can be timed.
3. **Candidate A — slope extrapolation.** Per trophy, logit(rate) is
   regressed on log(players) across the refresh states; the slope (the
   net of dilution and progression) is shrunk toward its tier mean and
   applied to the forecast player growth, anchored on the last rate.
4. **Candidate B — local velocity.** The freshest refresh-to-refresh
   logit moves give a per-hour velocity, tier-shrunk and damped toward
   zero over the horizon.
5. **Candidate C — cohort convolution (new in v3).** Cohort sizes are
   the increments of players_tracked across refresh states. Each cohort
   earns trophy i following a Weibull CDF in hours since joining,
   F_i(u) = 1 − exp(−(u/λ_i)^κ), with the shape κ shared across trophies
   and the scale λ_i fitted per trophy on the case's own history by
   matching predicted earners to observed earners, then regularized
   toward its tier's median λ. The forecast convolves the player-growth
   curve — past cohorts ageing, future cohorts arriving in twelve
   horizon slices — with F_i; the rate is forecast earners over forecast
   players. Fast trophies (small λ) outrun dilution; slow ones are
   diluted by every new cohort, exactly the split the live page shows.
6. **Self-calibration.** A rolling-origin backtest inside the case's own
   history truncates the states at several past checkpoints, re-runs the
   pipeline, and scores each candidate against the states that really
   followed. The residuals pick the A/B/C ensemble weights, a shrunk
   global bias shift, and the spread scale.
7. **Spread.** Each trophy's sigma mixes global, per-tier and per-trophy
   residual dispersion, grows with the square root of the horizon ratio
   between the backtest and the real forecast, and adds a refresh-step
   risk term. On top of that sits a **quantization-aware floor** (new in
   v3): rates print to 2 decimals, so below ~0.5% one print step is a
   large logit move (0.01 points at 0.05% is ~0.2 logit), and the spread
   never goes below what the printing grid alone implies. Quantiles are
   clamped to [0, 100] and every row is non-decreasing.
8. **Degenerate histories.** If a case has fewer than two distinct pages
   there is no structure to fit, so the forecast falls back to the last
   rates with a wide, dilution-aware asymmetric band: the median shifts
   down by the player growth an unseen refresh can hide, most strongly
   for rare trophies, and the lower side of the band is far wider than
   the upper side — the shape that carried the practice cases, whose
   histories are one flat page straddling the first refresh.

Any internal failure falls back per case to a wide random-walk spread
around the last read. The program never crashes, never emits a
non-monotone or out-of-range quantile, is deterministic, uses no
packages, and runs in a few hundred milliseconds per case.

## Run

    mkdir -p out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

Reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`), each holding five quantiles (0.05, 0.25, 0.5, 0.75, 0.95)
per trophy, in percent.

## License

MIT — see `LICENSE`.
