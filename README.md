# Model Citizens: multimodal Senate forecasting research

Open **reports/index.html** for the research report. It contains the current race review, scoring calculator, source checks, document evidence, and strategy comparisons.

This is an initial research system, **not a finished competition submission**. It uses publicly accessible sources and local computation. There are no paid data feeds, model API calls, registrations, or new service charges. Work performed in Codex or Claude continues to use your existing access and limits.

## Actual multimodal model, now trained

A local PyTorch network learns from **poll numbers, original PDF images, and report text**. It uses free frozen CLIP encoders and trains our own small forecasting head. We completed the first experiment on 20 historical training races, selected epochs on two 2022 races, and tested on three 2024 races. Training used no paid API or cloud GPU. All 26 source crops, including the unlabelled 2026 demonstration, received an AI-assisted visual check.

**The first result was negative:** the multimodal model scored worse than its matched numeric-only network (2024 Brier 0.592 versus 0.435; lower is better). All trained variants missed all three held-out winners. It remains experimental and does not change the competition CSV. This tiny sample cannot support an accuracy or winning claim. We preserve the failed comparison and saved weights.

Read [the model card](docs/MODEL_CARD.md), [reproduction instructions](docs/REPRODUCING.md), [experiment log](docs/EXPERIMENT_LOG.md), and [full measured results](reports/multimodal.json). The original statistical research pipeline below remains available separately. For the next version, see the ranked [improvement plan](docs/IMPROVEMENT_PLAN.md) and [image-input audit](reports/image-input-audit.json). The audit identifies a square-crop preprocessing defect in v0.1; its measured impact on forecast accuracy is not yet known.

## What is working

- All 35 race IDs from the organizer's Senate template are preserved.
- Free public polling feeds for 29 races are archived, with 381 raw poll rows in this snapshot.
- Market quotes, primary settlement rules, and source hashes are stored. Thirty-three race mappings pass the current state/year checks.
- A regularized fundamentals baseline uses prior presidential results and the president's party in midterms. Predictions for 145 eligible races in 2010–2018 train on earlier cycles only.
- A provisional poll average blends with the fundamentals margin. Four independent-candidate races use an explicitly labelled neutral prior and market-proxy minor-outcome probabilities.
- Unpolled races retain a labelled market probability proxy. The older fundamentals model has not earned promotion to precise present-day tail probabilities.
- Correct-pick rewards, expected-point selection, and comparisons against five hypothetical entry policies are implemented.
- A university poll's chart, overall table, and methodology are reconciled. Its original PDF's dates replace a conflicting tracker date. The poll enters once.
- An offline replay reproduces the research snapshot without a network call.

## Run it

Use Python 3.11 or later. The libraries listed in `requirements.txt` were already available when this project was built. No additional packages were needed for that original statistical pipeline. The new multimodal experiment uses the free packages in `requirements-training.txt`, installed in an isolated environment. On another computer, those open-source dependencies must be available before running.

From this project folder:

```sh
python3 run.py --as-of 2026-10-21
python3 -m unittest discover -s tests -v
```

The first command replays the cached snapshot offline in the original local workspace. A public clone includes numeric feed/market/returns snapshots, discovery metadata, and the hash-pinned reviewed PDF extraction. It replays offline using those facts. Publisher PDF/HTML originals and their images are omitted; restore them with `python3 scripts/restore_sources.py --all` for a fresh document audit. Mutable URLs can fail exact-hash restoration. Training from the included embeddings and saved-weight inference also work offline without publisher downloads. To collect new public captures, explicitly refresh using the actual current date:

```sh
python3 run.py --refresh --as-of YYYY-MM-DD
```

Replace `YYYY-MM-DD` with the forecast date in America/Los_Angeles. New data cannot establish availability on a past date. Original source captures are retained under `data/raw`, indexed in `data/sources.json`, and checked by SHA-256. When replaying an older date, the cache selects an eligible capture at or before that date.

The refreshed report is a local HTML file and needs no hosting subscription. It works by opening `reports/index.html`. Its controls and calculations run locally.

## Outputs

- `reports/index.html`: readable research report.
- `reports/research.json`: full model, evidence, and scenario results.
- `reports/validation.json`: every historical holdout prediction and training cutoff.
- `reports/submission-readiness.json`: reasons the draft is not ready.
- `drafts/predictions-RESEARCH-ONLY.csv`: an exploratory expected-points entry with the organizer's exact column names. Kentucky and Louisiana remain blank because their benchmark mappings are unresolved. This CSV is intentionally incomplete.

The draft party labels are `D`, `R`, and `I`. Confirm the organizers' required party codes before export for submission. Candidate names come from market metadata and still need a separate, current ballot-roster check. Alaska currently has generic party labels. Do not upload this research CSV as a finished entry.

## Forecasting and scoring are separate

`q` is the estimated chance of the selected outcome. `p` is the contest benchmark probability, which determines the reward. Current bid/ask midpoints are only **proxies** for the unavailable official November 1 benchmark.

Expected points for an outcome:

```text
q × -log2(max(p, 0.01))
```

Benchmark prices are never normalized in this calculation. For a labelled market-proxy forecast only, quotes are normalized across the included outcomes to construct a probability distribution. That distribution is not an independent estimate or the official benchmark.

The selected pick can have less than a 50% chance of winning. Its CSV probability remains the honest model probability. Confirm how the organizers interpret that probability column.

## Model settings and limits

The fundamentals baseline is ridge regression on prior presidential state lean, prior presidential national margin, and the president's party in a midterm. It excludes special elections, Louisiana's historical jungle-primary contests, same-party top-two races, uncontested contests, and races where the two major parties together earned less than 85% of candidate votes.

Historical source data ends in 2018. The 2024 presidential data provides the current state-lean feature. Current forecasts do not use 2026 election outcomes. Historical features use presidential results from strictly earlier years, although the underlying datasets have retrospective revisions. This is not a recreation of archived real-time data vintages.

The provisional polling component uses at most the most recent release per pollster, a 90-day eligibility window, a 21-day recency half-life, sample-size and voter-screen weights, and a 4-percentage-point shared-error floor with additional staleness and dispersion allowances. These settings have not completed historical calibration. They are research assumptions, not established confidence levels. No pollster reputation grades or subjective candidate impressions are used.

Poll-feed dates may not distinguish publication dates from field dates. Their first-seen capture timestamps bound availability. Verify original releases for the races driving the strategy. The university PDF correction has a verified field date and sample size.

Minor outcomes in independent-candidate races retain market-proxy probabilities, rather than invented model estimates. Independent-race priors and the Gaussian tails require particular scrutiny. Historical probabilities are diagnostics for the fundamentals baseline only; they do not validate the polling blend or the submission strategy.

The simulator compares hypothetical competitors under the provisional forecast, using three assumed latent correlations. Ties split first-place credit. Its first-place shares are conditional scenario diagnostics, **not probabilities of winning the real competition**. Candidate ordering for the copula and competitor choices are scenario assumptions.

## Multimodal work

The first document case is the University of Houston Hobby School's October 2026 Texas survey. The original PDF is cached, the relevant pages are rendered, and those images were visually checked in the building session. The chart and overall table both give 48/46/3/3. The PDF methodology specifies October 10–14, while a university-hosted tracker lists October 10–15.

The verified PDF hash is recorded in `config.json`. If a refresh changes its bytes, visual verification becomes false and the extraction is withheld as a primary-source correction until reviewed again. Text extraction and chart/table agreement alone do not establish that a new document has received visual review. `audit_document.py` prepares new local PDF evidence for review.

The genuine neural multimodal experiment is now trained, as described above. Its negative held-out result keeps image/text features out of contest picks. Subgroup figures are not extra independent observations. See `docs/MODEL_CARD.md` and `EXPERIMENTS.md`.

## Remaining work, in priority order

1. Independently verify the current ballot roster and repair the Kentucky/Louisiana market mapping. One returned `SENATELA-26` event describes Kentucky in its title and settlement rules. It is quarantined for Louisiana. The returned Kentucky series has no 2026 markets. Do not fix either by trusting ticker letters alone.
2. Add primary-source verification for the polls that drive the largest score decisions. Resolve field/publication dates and candidate identity, and check older releases against the current matchup.
3. Extend the full statistical polling-blend training/holdouts through 2020, 2022, and 2024, with correct prior-election features. The separate small neural experiment now covers those cycles, but does not validate the full blend. Calibrate the complete polling model and long-shot tails. Free official results are available from the FEC and state election offices, but are not yet integrated here.
4. Expand the audited document dataset beyond the first 20 training races and test a new preregistered experiment. The initial multimodal comparison failed; keep its result visible. Keep the same baseline and forecast cutoff in each comparison.
5. Compare robust first-place strategies under several plausible model and competitor scenarios. Do not tune exclusively against one favorable assumed field.
6. Load the exact official November 1 weights when provided, confirm the probability-column rules, and regenerate the final entry before the November 2 deadline. The form did not specify a timezone.

## Sources and attribution

- Organizer template: https://docs.google.com/spreadsheets/d/1VhBefuanDtrB96Qh-_XwU5Y91mZFwhV2fRaM3-UtFdQ/edit
- MIT Election Data and Science Lab historical returns: https://github.com/MEDSL/constituency-returns
- MIT Election Data and Science Lab 2024 returns: https://github.com/MEDSL/2024-elections-official
- On Point Politics public polling feeds: https://www.tpsielections.com/polls/senate — these are a discovery and ingestion source, not verified individual poll releases.
- Kalshi public market data: https://docs.kalshi.com/getting_started/quick_start_market_data
- University of Houston Hobby School of Public Affairs, October 2026, “Texas Votes 2026: Statewide Races, Favorability & Election Administration”: https://www.uh.edu/hobby/texas2026/statewide26.pdf
- Texas Politics Project, “Texas 2026 U.S. Senate Poll Tracker,” University of Texas at Austin: https://texaspolitics.utexas.edu/blog/texas-2026-u-s-senate-poll-tracker
- FEC official past-results directory for further work: https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/

Per-source URLs, capture times, and hashes are in `data/sources.json`. Sources retain their original authorship and terms. Publisher originals and encoder weights are excluded from Git; source manifests, derived embeddings, our checkpoints, and reproducibility instructions are included. The user has authorized a public GitHub repository; publication status is recorded in `docs/PUBLICATION.md`. No organizer message or contest submission has been sent.
