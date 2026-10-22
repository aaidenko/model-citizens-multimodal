# Model Citizens multimodal experiment v0.1

This is a **trained neural network**, with real image and text inputs. It is an experimental prototype and has **not earned use in the contest forecasts**. The first held-out comparison made the multimodal network worse than its numeric-only counterpart.

## What is trained

Original poll PDF crop → frozen CLIP image encoder → 512 image features → 4 training-fitted principal components.

Text extracted from that same crop → frozen CLIP text encoder → 512 text features → 4 training-fitted principal components.

These eight features join four numeric features: two-party poll margin, remaining share, days until the election, and a likely-voter indicator. A small neural network with one eight-unit `tanh` hidden layer learns a correction to the poll margin. The correction is bounded to ±6 percentage points. The full forecasting head has **113 trainable parameters per seed**, with three seeds averaged. Small capacity is intentional given only 20 independent training outcomes. The likely-voter indicator happens to be constant in this initial sample.

The 151,277,313 encoder parameters are frozen. This is transfer learning: the forecasting head is trained by gradient descent; the foundation encoders are not trained from scratch. Text and image masking changes predictions, confirming that the learned network uses both inputs. That does not establish that either input adds predictive skill.

Encoder: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The model card dates the original model to January 2021. This places 2022 and 2024 outcomes after the encoder's stated creation date. Older training outcomes may be represented in the original pretraining corpus; these older races are not presented as prospective encoder evaluations. The downloaded weights' SHA-256 is recorded in `data/multimodal/embeddings-manifest.json`.

## Data

There are **26 reviewed document cases**: 20 training races in 2014–2020, two 2022 validation races (CT and NY), three 2024 test races (MI, PA, WI), and one unlabelled 2026 Texas demonstration.

The historical research pool contains 71 Quinnipiac PDF files plus a Marist New York report. Discovery selects August–November Senate-titled releases, excluding generic control/impeachment stories. Eligible historical snapshots were published at least 14 and at most 100 days before Election Day. The dataset freezes one latest usable, parsed, visually reviewed snapshot per race from that pool. This is a **convenience sample**, not complete coverage of those elections or every eligible poll. Some documents and matchups do not pass the parser. Do not describe this as a comprehensive backtest.

Images are crops of the original reports, not generated charts. They contain poll questions, toplines, and often cross-tab tables or earlier trends. Those extra numbers are parts of the same poll document, not additional independent outcomes. Numeric features use the reviewed major-party topline. No candidate photograph, demographic prediction about individuals, campaign ad, social post, or voter-level record is used.

The Marist New York case uses the **definitely voting** row, 52/39, not its registered-voter row of 54/34. The 2016 Patrick Murphy case is Florida; the 2018 Chris Murphy case is Connecticut. The name collision was resolved before fitting.

Labels use Democratic minus Republican votes divided by the two major candidates' combined votes. Historical MEDSL returns combine a candidate's fusion-ballot lines. The 2020 labels use FEC general-election totals; 2022 uses combined-party totals. MEDSL's archived 2024 state file is marked unofficial; these are retrospective labels, not independently reconciled certified returns. Runoffs, multi-candidate special races, and independent-led contests are not covered by this prototype.

`cases.json` records each source URL, source hash, page, exact crop, candidate identities, numerical extraction, publication date, election date, and outcome provenance. `visual-reviews.json` records Codex's AI-assisted visual inspections; these are not user or independent human sign-offs. A changed source hash requires a new review.

The Texas demonstration's October 21 availability date is a first-seen capture, not a claim about its original publication day. It has no outcome label and never enters training or validation.

## Temporal separation and training

- Train: 2014, 2016, 2018, 2020. One independent outcome per race.
- Validation: 2022, two races. Select 50, 150, or 300 epochs by race-level margin RMSE.
- Test: 2024, three races. Opened after the configuration was written; no tuning against these results.
- Current inference: 2026 Texas, using saved weights and preprocessing.

Numeric standardization and both four-component PCA transformations fit on training rows only. Text is embedded in chunks of up to 36 words, capped at 288 words and 77 model tokens per chunk; normalized chunk embeddings are averaged. Images are rendered at 120 dpi and resized/cropped by the pinned CLIP processor. This limits fine table reading and is a likely weakness. A subsequent input audit found a concrete preprocessing defect: the square center crop removes candidate names and topline cells in some wide tables. See `IMPROVEMENT_PLAN.md` and `reports/image-input-audit.json`. The original experiment is retained for reproducibility; no forecast improvement from a repair has been measured.

Optimization uses weighted margin MSE, AdamW, learning rate 0.01, weight decay 0.03, and seeds 17, 29, 43. The code retains inverse-race weighting if later datasets contain repeated polls; this initial frozen dataset contains one poll per race. All variants use the same examples and epoch grid.

For diagnostic probabilities only, `q_D = NormalCDF(predicted_two_party_margin / 6)`. **Six points is an assumed error scale, not a fitted calibration result.** These probabilities are conditional on the major-party comparison and are not complete multicandidate probabilities. No probability from this model is inserted into the contest CSV.

## First results

Lower margin RMSE and lower Brier score are better. Winner accuracy uses `q_D >= 0.5` to choose D; therefore an exact 50/50 Michigan poll receives a deterministic D tie-break.

| Model | 2024 margin RMSE | 2024 Brier | Winners correct |
|---|---:|---:|---:|
| Uncorrected poll | 5.394 | 0.386 | 2/3 |
| Numeric neural network | 3.180 | 0.435 | 0/3 |
| Numeric + text | 4.567 | 0.542 | 0/3 |
| Numeric + image | 4.168 | 0.505 | 0/3 |
| Numeric + text + image | 5.238 | 0.592 | 0/3 |

The full model is worse than the numeric network on both measures. The numeric network improves margin RMSE versus the raw poll while worsening winner probability scoring. This illustrates why a lower margin error alone does not establish useful contest probabilities. Even the two validation races are both comfortable Democratic wins and provide little calibration information about close races.

Image masking changes the held-out forecast margin by an average 1.137 points; text masking changes it by 0.988 points. These are input sensitivity diagnostics, not causal attribution or evidence of improvement.

The JSON report retains every race prediction, validation epoch comparison, sampled training losses, seeds, runtime, parameter changes, and artifact hashes. All failed comparisons are retained. The current Texas output is an inference demonstration under a new publisher/layout, not a recommended forecast.

## Intended use and next decision

Use this implementation to reproduce the experiment, expand audited historical coverage, and test additional signals. Do not rely on it for precise long-shot probabilities or claim it can win the competition. The initial result does not show that multimodal forecasting cannot work; it shows this model and tiny dataset have not demonstrated an advantage.

The next experiment should add many more independent races, several publishers/layouts, more close and Republican-winning validation cases, and a new genuinely untouched evaluation set. The 2024 set is now consumed for future model-development purposes. Keep the original result as a historical benchmark; do not repeatedly tune to those three outcomes. A promotion check needs at least 20 independent held-out races, improvement against the same numeric baseline on proper probability scores, and a defensible calibration of close-race and tail probabilities.

## Costs and publication

All training ran on the local CPU. The forecasting-head fitting took under a second in the recorded run; source preparation and frozen-encoder inference take longer. This is not a claim that the entire project took under a second. No hosted inference endpoint, paid GPU, paid feed, API key, or new subscription was used. Downloads and package installation were free; existing Codex access remains subject to the user's existing plan and limits.

Original publisher PDFs, HTML, rendered page images, virtual environments, and the large external encoder weights are omitted from Git. Their URLs and hashes are included. Included embeddings and our small model checkpoints permit offline retraining and inference. Publisher originals retain their authorship and terms. See `THIRD_PARTY.md` and `REPRODUCING.md`.
# Preserved version notice

This card describes the original v0.1 experiment. The completed versioned improvements and their measured limits are in [V2_MODEL_CARD.md](V2_MODEL_CARD.md). Original v0.1 features, weights and negative results remain unchanged.
