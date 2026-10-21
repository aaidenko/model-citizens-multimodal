# v0.2 evaluation protocol

Fixed on October 21, 2026 before computing v0.2 test scores. This is a retrospective experiment with revised source archives, known election history, and consumed v0.1 evidence; it is not a blind prospective prediction contest.

## Population and inputs

Senate general elections, 1998–2024, with one Democratic and one Republican nominee and valid two-party polls. Exclude Louisiana jungle elections, runoff elections, same-party contests, independent-led races, and special elections where their identities cannot be resolved. Forecast horizon: 14 days before election. Historical 538 median field dates receive a seven-day availability embargo; this is a conservative proxy, not proof of publication. The 2024 archive requires both completed fieldwork and an ingestion date strictly before the forecast date. Poll window: 90 days. Latest eligible question per poll and latest release per pollster, with date sorting explicit. Population preference LV, RV, adult; do not treat repeated questions as independent polls.

Whitelist: poll shares, dates, sample size, population/mode/partisan indicators, pollster identity for deduplication, and strictly previous presidential state/national results. No retrospective pollster grades, actual outcomes, error measures, winner identity, or current-cycle presidential result may enter features. Labels are stored separately. Previous result vintage is checked on every row. The ratings archive's top-two selection and imputed sample sizes remain limitations. Numeric and reviewed-document race counts are reported separately.

## Temporal evaluation

- Numeric model selection: train through 2016, development 2018 and 2020. Ridge residual alpha in {4, 40, 400}; neural residual head with 8 hidden units, tanh, ±6-point cap, AdamW, epochs in {100, 300}, seeds {17, 29, 43}. Select by development margin RMSE; ties prefer simpler settings. Refit the selected settings through 2020.
- Document head: train through 2018, select on 2020, refit through 2020. If no development examples exist, use zero document contribution. Frozen January 2021 CLIP, train-only PCA with at most four components per modality, regularized small head, ±3-point cap. Candidate blend weights {0, .25, .5, 1}, epochs {100, 300}; select by development margin RMSE. The shared baseline and numeric document facts are identical in all four arms. The encoder was unavailable before 2021, so earlier cycles are retrospective representation training/development only.
- Calibration: 2022, after model selection/refit. Fit a two-parameter Student-t(df=5) margin-error location/scale to numeric baseline residuals. Apply the same calibration to all matched document arms. Estimate between-cycle error from earlier development residuals for a separately labelled correlation diagnostic. Calibration-set scores are in-sample diagnostics, not independent tests.
- New test: eligible 2024 races except Michigan, Pennsylvania and Wisconsin (already consumed by v0.1). No test-driven architecture, feature, blend, calibration or epoch changes. Preserve all results even if the direct multimodal branch loses. Report the three consumed races separately if evaluated at all.
- Publisher holdout: Monmouth/Emerson documents are never used to fit the document head or its PCA. Score eligible unseen-publisher documents in 2024 separately; do not mix a publisher test with development. If no eligible reviewed examples survive, report the missing evaluation rather than invent coverage.

## Image/document processing

Preserve the full reviewed original crop with aspect-ratio padding. Add square overlapping tiles so wide tables retain readable name and value columns. Disable CLIP's additional resize/center crop for already prepared 224×224 inputs. Audit actual encoder inputs. Native PDF extraction first; local OCR fallback is checked against reviewed source cells and is not allowed to auto-approve a source. Documents, crop coordinates, source hashes, review methods, exclusions and missingness are versioned. Tiles/pages/subgroups never increase independent outcome counts. Publisher originals are not redistributed in Git.

## Reporting and promotion

Primary probability measures: Brier and log loss; also accuracy, margin RMSE, reliability bins and paired per-race changes. Four matched arms: numeric document facts, +text, +image, +both. Include the expanded numeric-only test separately. Small document samples and shared election-cycle error prohibit a claim that a measured difference guarantees future improvement. Allow weight zero and keep v0.1 unchanged. Neither model is automatically promoted into the contest CSV. The official benchmark, roster checks and unresolved rules remain separate submission blockers.

New service spend: $0. Local CPU, open-source packages and frozen encoders only. No paid inference, cloud GPU, automated organizer contact or contest submission.
