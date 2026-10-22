# v0.2 model card

Built locally on October 21, 2026. **Research only; not promoted into the competition CSV.** Public repository: https://github.com/aaidenko/model-citizens-multimodal.

## What improved

The implementation now preserves the complete source image, adds readable overlapping tiles, trains against a much larger numeric history, fits uncertainty on separate earlier races, and allows validation to suppress an unhelpful document correction. Actual saved neural weights, forward predictions, source hashes, negative results, and offline reproduction code are included. New service spend was **$0**.

The larger selected numeric model reduced test margin RMSE from **5.69 to 4.76 percentage points** and Brier score from **0.02096 to 0.01510**, compared with uncorrected polling averages using the same fitted probability conversion. These are 25 newly selected 2024 races, including many safe seats. Michigan, Pennsylvania, and Wisconsin were excluded because v0.1 already consumed them. The result is not comparable to v0.1's three close-race test, a blind prospective evaluation, or a contest-winning probability.

**The direct multimodal correction did not earn inclusion.** Earlier-cycle validation selected weight zero for both the image and full multimodal arms. Their 14-race test scores equal the expanded numeric baseline. Improvement against the separately trained numeric document head is caused by suppressing its correction, not evidence that images predict election outcomes better.

## Data and split accounting

| Dataset | Initial training | Development | Final fit | Calibration | New test | Consumed/excluded test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Numeric snapshots, 346 races | 249 through 2016 | 45 in 2018/2020 | 294 through 2020 | 24 in 2022 | 25 in 2024 | 3 |
| Reviewed documents, 50 races | 29 through 2018 | 5 in 2020 | 34 through 2020 | 2 source cases; shared numeric calibrator | 14 in 2024 | 0 |

The numeric snapshots use 1,802 selected poll observations, with at most one latest eligible release per pollster and one race-level target. More pages, questions, subgroups, image tiles, or repeated polls never increase independent outcome counts. The reviewed document set contains 36 source cases from publisher PDFs and 14 screenshots of original publisher HTML tables. Twenty-two exact v0.1 source reviews were inherited; 28 new source cases received visual checks. Publisher/layout coverage is still a convenience sample; this is not a 100-race multimodal training dataset.

The numeric development set includes 27 Democratic and 18 Republican wins. Document training includes both parties (16 D/13 R initially, 16 D/18 R after refit), but **all five document-development races were Republican wins**. That narrow development set remains a serious limitation; expanding its party/cycle coverage was not achieved by this collection. It does not by itself prove the cause of an error. The class counts are explicit in `reports/v2-cohort-audit.json`, and the now-consumed test must not be used to choose a replacement validation split.

Sources: FiveThirtyEight's public historical ratings archive and its archived 2024 long-form poll CSV; MEDSL historical and 2024 returns; original Quinnipiac, Marist, university, media, and pollster releases. URLs, captures, hashes and exclusions are included. FiveThirtyEight-derived data retains CC BY 4.0 attribution; our code license does not relicense those data or publisher materials.

The forecast horizon is 14 days before election, with a 90-day poll window. Historical median field dates receive a seven-day release embargo **as an availability proxy**. It cannot establish actual publication time. The 2024 feed requires completed fieldwork and an ingestion date strictly before the forecast date, with a one-day timezone guard. Dated original documents are checked separately. Revised archives, retrospective top-two selection, and archive-imputed sample sizes remain limitations. Some MEDSL 2024 rows are marked unofficial.

Excluded cases include ineligible/ambiguous special elections, runoff-dependent contests, Louisiana jungle elections, independent-led and same-party contests. This model estimates conditional D/R outcomes. It does not validate the four independent-candidate races in the current contest roster.

## Models and selection

The numeric forecaster uses the polling average, remaining share, dispersion/staleness, poll count/sample size, missingness, partisan/mode indicators, strictly earlier presidential state/national lean, a midterm indicator, and a **returning-winner proxy**. The proxy matches the current nominee to the winner six years earlier; it misses appointments and is not a fully verified incumbency roster. No candidate-quality or historical generic-ballot feature was invented to fill gaps. Retrospective ratings, accuracy measures and current-cycle outcomes are excluded from features.

Development compared ridge residual alpha 4/40/400 and actual PyTorch residual networks at 100/300 epochs, with seeds 17/29/43 and a six-point cap. Ridge alpha 4 won the prespecified development margin-error criterion; a separately trained neural comparator is also retained. Test scores do not choose the winner. Refit ends in 2020. Forward numeric predictions for earlier years train the document correction, avoiding in-sample baseline fits as its targets.

Document arms share the same baseline and numeric document facts. Frozen January 2021 CLIP ViT-B/32 supplies text/image embeddings. Train-only PCA retains at most four dimensions per modality. A small 8-unit tanh network learns a correction capped at three points. Development selects epochs 100/300 and blend weights 0/.25/.5/1, then refits through 2020. The four arms have 73/105/105/137 trained parameters per seed. Constant training features are disabled so novel holdout values cannot activate untrained weights. All heads record nonzero parameter updates.

Student-t(df=5) margin errors are fitted to 24 held-out 2022 numeric residuals: location approximately +0.428 points, scale 5.589 points. The same conversion is used for all document arms. This replaces the assumed six-point scale; it does not establish calibrated extreme tails. Calibration-set scores are fitting diagnostics. Shared-cycle error is described separately using only two development-cycle means, insufficient for a reliable correlation estimate.

## Matched document test: 14 races

| Arm | Selected document weight | Brier ↓ | Margin RMSE ↓ | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Expanded numeric baseline | — | 0.02404 | 4.41 | 14/14 |
| Numeric document facts | 1.00 | 0.04700 | 5.15 | 13/14 |
| Numeric + text | 0.25 | 0.02644 | 4.40 | 14/14 |
| Numeric + image | 0.00 | 0.02404 | 4.41 | 14/14 |
| Full multimodal | 0.00 | 0.02404 | 4.41 | 14/14 |

Unblended scores, reliability bins, per-race predictions, actual parameter changes, and modality masking diagnostics remain in `reports/v2-documents.json`. Never describe zero-weight image results as a demonstrated image benefit. The prespecified Monmouth/Emerson PDF holdout had no eligible retrieved source and is **unavailable**. All 14 new document-test races use publishers absent from head training; the additional diagnostic is confounded with election cycle and does not isolate layout effects.

## Reader and image checks

Every source view includes a complete aspect-preserving padded image plus overlapping readable tiles. Additional CLIP center cropping is disabled. All 50 first views retain the complete reviewed source; the encoder receives 5–16 views per case. Inverse-normalized actual encoder inputs were visually checked for Colorado's previously lost name/56/38 columns, the corrected Texas page 89, and a browser table.

Native PDF/DOM extraction is preferred. The optional local RapidOCR/ONNX reader recovered 8/10 candidate cells across five reviewed cases from preserved full-resolution sources, versus 0/10 from old center-cropped inputs. The candidate-as-column Texas table failed and remains a failure. This tiny diagnostic compares reader inputs, not new encoder tiles or general OCR accuracy. Its newer OCR model is not a historically available forecasting encoder and its outputs are not model features or automatic visual approvals.

Source checks corrected the Texas table-of-contents selection, a partial West Virginia chart crop, and Tennessee's document-level likely-voter denominator (954 rather than feed 971). Rhode Island's overall adult sample was not substituted for its registered-voter denominator. Question variants and leaner/full-ballot distinctions remain explicit. Reviewed document facts enter all matched arms equally; these corrections were not selected by test outcome.

## Limits and use

One test cycle, small document samples, nonrandom source coverage, partially proxied release dates, revised archives, known historical outcomes, new layouts, and sparse tail validation prevent a winning or general-accuracy guarantee. CLIP was unavailable before 2021: older document cycles are retrospective representation training/development, not reconstructed pre-2021 real-time forecasts. Currently rendered legacy HTML layouts also postdate the election; modern news/navigation are excluded from inputs.

The preserved v0.1 experiment remains negative and unchanged. The current 2026 statistical research CSV keeps its original provisional model. Official November 1 weights, current roster checks, unresolved Kentucky/Louisiana mappings, probability-column rules, and deadline timezone remain submission blockers. Forecast probability `q` and official benchmark reward probability `p` stay separate. No trading, organizer contact, paid inference, rented GPU, or competition submission occurred.
