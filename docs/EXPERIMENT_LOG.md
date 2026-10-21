# Experiment log

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
