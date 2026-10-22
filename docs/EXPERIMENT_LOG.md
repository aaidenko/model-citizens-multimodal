# Experiment log

## 2026-10-21: expanded v0.2 experiment

**Protocol:** numeric temporal selection was committed as `ed0094a` before numeric test scoring. Source-availability amendments were committed as `68a549f` before document test scoring. The original v0.1 Michigan/Pennsylvania/Wisconsin test races are consumed and excluded from the new test. This remains retrospective research with known history, revised archives and partially proxied publication times.

**Data:** 346 numeric snapshots, 1,802 selected poll observations, 50 reviewed document races. Initial numeric fit: 249 races through 2016; development: 45 in 2018/2020; final numeric fit: 294 through 2020; calibration: 24 in 2022; new test: 25 in 2024. Document fitting grows from 20 to 34 races, with 29 earlier training and five development races; two 2022 source checks and 14 new test races remain separate. Fourteen added older cases use screenshots of original publisher-rendered HTML tables, rather than invented charts. The total is not 100 multimodal training races.

**Changes:** full-image padding plus overlapping tiles; no additional encoder center crop; train-only PCA/scaling with constant-feature suppression; poll averaging and strictly prior state lean/returning-winner features; actual small PyTorch residual heads; development-selected bounded document contribution including zero; Student-t error calibration fitted in 2022. Original-source review fixed a table-of-contents page selection, clipped chart, population/denominator mismatches, and forced-choice/leaner distinctions. Historical median field dates are explicitly labelled as median anchors, not fieldwork end dates; clarifying that schema left the numeric feature file byte-identical.

**Numeric result:** ridge alpha 4 won the predefined development margin criterion. New-test margin RMSE 4.7644 versus raw-poll 5.6874; Brier 0.015104 versus 0.020963 with the same fitted conversion. A retained neural comparator has Brier 0.014774 and margin RMSE 5.2144; it does not replace the selected model after seeing test scores. All 25 selected-model winners were correct, but the cohort includes many safe races and excludes three previously consumed close races. It cannot establish future 100% accuracy or winning odds.

**Matched document result:** baseline Brier 0.024036; trained numeric-document head 0.046996 (weight 1); +text 0.026442 (weight .25); +image and full multimodal 0.024036 (both weight zero). Full multimodal margin RMSE 4.4132 therefore equals the baseline. The nonzero neural parameter changes and unblended/modal-masking diagnostics are retained. There is no demonstrated image benefit from a branch suppressed by validation.

**Source availability:** PPP access was blocked by network policy and was not bypassed. No eligible Monmouth/Emerson PDF survived the collection rule, so that prespecified holdout is unavailable. The secondary diagnostic includes 14 new-test races with publishers absent from head training; it is confounded with cycle. A five-case local OCR diagnostic recovered 8/10 cells from preserved full-resolution sources versus 0/10 from old center crops; the candidate-as-column Texas case failed. OCR never grants visual approval or feeds these benchmark predictions into forecasts.

**Coverage audit:** numeric development contains 27 D/18 R wins. Document training is 16 D/13 R before development and 16 D/18 R after refit; its five development outcomes are all R. Broader document-development party coverage remains unfinished. This strengthens the reason to withhold promotion and to require a separately planned future experiment, rather than tune the consumed test.

**Decision:** retain the stronger numeric research baseline and the actual trained document heads; do not promote the direct multimodal correction or change the current 2026 CSV. Preserve v0.1 bytes and its negative result. The separate fitted uncertainty model does not validate the original 2026 blend's tails or independent-candidate handling. Detailed results: `reports/v2-numeric.json`, `v2-documents.json`, `v2-image-audit.json`, and `v2-ocr.json`.

**Validation and cost:** local CPU only; free packages and weights; no new service spend. All 43 tests pass in the training environment. Exact clean-public-copy reproduction and artifact comparisons are recorded separately in `reports/v2-reproducibility.json`. Local HTML resources/structure and the scientific result figure were checked; the in-app browser refused local-file preview, so no completed browser-render inspection is claimed for the new report section. No organizer contact or contest submission occurred.

## 2026-10-21: document fusion v0.1

**Question:** Does a trained forecasting head improve on the same numeric-only head when supplied the text and original PDF images of historical poll tables?

**Configuration fixed before reading test scores:** frozen January 2021 CLIP ViT-B/32, four image PCA components, four text PCA components, four numeric features, eight-unit neural hidden layer, ±6-point correction, seeds 17/29/43, epochs selected from 50/150/300 using 2022 margin RMSE. A 6-point normal error scale gives diagnostic probabilities. This scale was not calibrated.

**Dataset:** Quinnipiac original poll PDFs plus one Marist validation case. One latest usable, parsed, reviewed source per race in a 14–100-day pre-election window. Twenty training races in 2014–2020, two validation races in 2022, three test races in 2024. One unlabelled 2026 Texas document is for saved-weight inference only. This is a small convenience sample.

**Checks before training:** extracted toplines reconciled to rendered original crops; publication dates came from the publisher's release archive; candidate/state identities checked; Patrick Murphy (FL 2016) and Chris Murphy (CT 2018) disambiguated; Marist NY definitely-voting row selected; source hashes frozen; no 2026 outcomes. FEC 2020 and 2022 source pages were visually checked for manually curated vote labels.

**Result:** numeric-network test Brier 0.434580; numeric+text 0.541934; numeric+image 0.505394; full multimodal 0.591515. Lower is better. The uncorrected poll benchmark is 0.386472. All trained variants missed all three close-race winners. The full model's test margin RMSE is 5.237955 points versus 3.179584 for the numeric head.

**Decision:** do not promote the model. More inputs did not earn improved predictive performance. Keep every result and checkpoint. The tiny, unbalanced validation/test sets cannot establish skill, and the 2024 set is now consumed for subsequent development. Do not tune until those three predictions look good.

**Actual computation:** local CPU, free downloaded packages and encoder weights, no paid API or rented GPU. Forecast-head optimization took 0.799 seconds in the initial recorded run; document preparation and encoding took additional time. Full timing, parameter changes, loss samples, seeds, predictions, and hashes are in `models/multimodal/run.json`. Small floating point differences are expected in independently recreated encodings or runtimes.

**Next experiment:** expand independent historical races and publisher layouts, include close-race and Republican-winning validation cases, compare a polling average plus fundamentals against all document variants, and reserve an untouched evaluation set. Calibrate probabilities separately from margin prediction. Until then, use the existing contest-scoring research workflow and audited poll facts without applying this network's adjustments.

## Earlier statistical baseline

The original fundamentals ridge model used earlier-cycle presidential lean and midterm indicators, with forward holdouts in 2010–2018. Its 145-race Brier score of 0.145406 is **not comparable** to the three close-race document holdout scores above. It does not validate the current polling blend or the neural experiment. See `reports/validation.json` and the main README.
