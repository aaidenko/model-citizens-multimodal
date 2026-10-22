# Reproduce the trained experiments

Use Python 3.13 for the recorded environment (the baseline also supports Python 3.11+). All dependencies are free. No account, payment method, API key, inference endpoint, or rented GPU is required. Commands below run from the repository root. On another operating system, use the platform's official CPU PyTorch distribution if the pinned wheel is unavailable; record the changed runtime before comparing artifacts.

## Version 0.2: offline reconstruction, training and inference

Install the free environment once, then run from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python scripts/build_v2_data.py
.venv/bin/python train_v2.py
.venv/bin/python train_v2.py --documents
.venv/bin/python predict_v2.py 2024-OH-the-washington-post
.venv/bin/python run.py --as-of 2026-10-21
.venv/bin/python -m unittest discover -s tests -v
```

The same offline reconstruction, training, replay and test sequence is available as `.venv/bin/python scripts/verify_v2.py --retrain`. Without `--retrain` it only replays the report and runs tests.

After package installation, this sequence uses no network, API key, paid service, or external encoder download. It reconstructs numeric snapshots from the included CC BY 4.0 CSV captures, trains the numeric selection/calibration, trains the four document heads from included frozen embeddings, and performs saved-weight inference. Model selection uses development years; repeated reproduction does not create a fresh holdout. Run reports record timings, which can change without changing predictions or model weights.

`data/v2/cases.json` stores reviewed source identities/facts without targets; `labels.json` stores targets separately. `embeddings-manifest.json` pins source geometry, case order, encoder revision and embedding hash. `models/v2/` contains numeric coefficients, calibrated uncertainty, small neural checkpoints and train-only document preprocessing. `reports/v2-*.json` preserve complete measured results, including zero-weight decisions and unblended diagnostic scores.

The saved inference command loads weights and training preprocessing only. Its unblended neural margin is diagnostic; a selected document weight of zero produces the numeric baseline prediction. Training-case inference is labelled as a fit, not an independent forecast. The v0.2 inference example is historical; it is not a current contest entry.

To restore publisher originals explicitly:

```sh
.venv/bin/python scripts/restore_v2_sources.py --all
```

Exact-hash restoration can fail on mutable or unavailable URLs. Changed bytes are quarantined and do not inherit approval. PDF originals can then be rendered again. The 14 HTML cases also require the original publisher table to be rendered in a browser, its date/candidate/population checked, and its screenshot/text capture compared against the recorded hashes and crop. They are currently rendered legacy pages, not archived historical layouts. The source collection tools prepare candidates, not approvals; a new visual review is necessary for changed inputs. Do not recreate missing screenshots by drawing a new chart or rendering election outcomes.

`scripts/collect_v2_documents.py` gathers the defined public publisher PDFs. `--ppp` is disabled by default; that archive was blocked in this environment and was not bypassed. `scripts/discover_v2_html.py` prepares the browser-review plan for additional dated Quinnipiac releases. `scripts/prepare_v2_review.py` and `scripts/assemble_v2_cases.py` use locally reviewed sources and hash-specific review records; they are not necessary for offline training from included features. Source preparation needs original bytes, the ignored local review plan and fresh browser captures; those third-party originals are deliberately absent from a public clone.

With exact reviewed originals and the same local CLIP snapshot, optional re-embedding is:

```sh
.venv/bin/python scripts/embed_v2_documents.py --encoder-path LOCAL_SNAPSHOT_DIRECTORY
```

For optional local OCR, install `requirements-ocr.txt` in the environment and run `scripts/benchmark_v2_ocr.py --encoder-path LOCAL_SNAPSHOT_DIRECTORY`. This uses bundled local OCR models, never hosted inference. The benchmark's five cases and unsupported column-layout failure are retained. It does not feed OCR predictions into forecasting or auto-approve sources. Optional figure regeneration uses Matplotlib with `scripts/plot_v2_results.py`; the finished figure is included, so plotting is not required for model reproduction.

## Preserved v0.1 training from included derived features

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python train_multimodal.py
.venv/bin/python predict_multimodal.py --case 2026-TX-2026-10-21
.venv/bin/python -m unittest discover -s tests -v
```

Package installation needs internet; after installation the training and prediction commands do not. `embeddings.npz` contains frozen encoder outputs from the reviewed original images/text. These derived features, curated labels, and saved small checkpoints are included. The original numeric polling/market/returns snapshots and reviewed PDF extraction also support an offline baseline replay from a clean public copy. Training records the hash of that exact dataset, all seeds, epoch candidates, selected epochs, and resulting artifact hashes. Tests and reports do not fetch data implicitly.

The `train_multimodal.py` command retrains all four predefined input variants and writes `reports/multimodal.json` and `models/multimodal/run.json`. The 2024 scores are an already consumed research benchmark, not a fresh holdout on each rerun. Do not optimize repeatedly against them.

`predict_multimodal.py` loads saved weights and training-fitted preprocessing, verifies hashes and case order, and predicts without refitting. The Texas result is illustrative, uncalibrated, conditional on the major parties, and not promoted to the contest forecast.

## Recreate image and text features from publisher originals

```sh
.venv/bin/python scripts/restore_sources.py
.venv/bin/python scripts/download_encoder.py --cache-dir work/encoder-cache
```

The encoder downloader prints a local snapshot directory. Pass that directory explicitly:

```sh
.venv/bin/python train_multimodal.py --encode --encoder-path LOCAL_SNAPSHOT_DIRECTORY
```

Replace the capitalized placeholder with the printed path. The external encoder is approximately 600 MB. The source archive is tens of MB. Both are one-time free downloads; neither is uploaded into the public repository. The encoder is pinned by revision and its downloaded weight hash is recorded. The encoding function requires a local directory and `local_files_only=True`, so it has no hidden online inference fallback.

Source restoration checks recorded SHA-256 hashes. If an original URL changes or becomes unavailable, it fails visibly; it does not silently substitute a new document into the old experiment. Re-review changed originals before creating new features. Recreating embeddings may vary slightly across library versions or hardware; our checked-in embeddings fix the initial experiment's inputs.

`scripts/collect_multimodal.py --discover` is for a **new research collection**, not historical snapshot reproduction. It discovers actual links from Quinnipiac's public archive and adds source records. Discovery candidates never automatically become reviewed training cases. The initial dataset was curated from those candidates; `cases.json` freezes the selected source/crop/topline for each example.

## Statistical research report

```sh
python3 run.py --as-of 2026-10-21
python3 -m unittest discover -s tests -v
```

This is separate from neural training. Numeric feed, market, and election-return snapshots are included. When publisher originals are absent, the offline replay uses our discovery metadata plus the previously reviewed PDF extraction, pinned to the original source hash. This replays the original review rather than claiming a fresh image inspection. Publisher PDFs/HTML and their images are omitted from public Git. `scripts/restore_sources.py --all` attempts exact-hash restoration of those originals. Some market and polling URLs are mutable; exact restoration can fail. In that case use the included numerical research outputs, or run a newly dated refresh and retain its new source manifest. A fresh download cannot prove that it was available on a past date.

The neural experiment's checked-in features, labels, and model weights do not depend on those mutable market URLs. The statistical pipeline does not import PyTorch and never starts training implicitly.

## What each file preserves

| Artifact | Purpose |
|---|---|
| `data/multimodal/cases.json` | Frozen examples, labels, identities, dates, page crops, source hashes |
| `data/multimodal/visual-reviews.json` | Hash-specific AI-assisted review record |
| `data/multimodal/sources.json` | Original report URLs, capture times, hashes |
| `data/multimodal/labels*.json` | Outcome labels and FEC source provenance |
| `data/multimodal/embeddings.npz` | Actual image/text encoder features, not synthetic examples |
| `data/multimodal/embeddings-manifest.json` | Feature/source/encoder hashes and preprocessing record |
| `models/multimodal/*.safetensors` | Small learned forecasting weights for four variants × three seeds |
| `models/multimodal/preprocessing.npz` | Scaling/PCA fit on earlier training years only |
| `models/multimodal/run.json` | Full measured experiment and artifact checksums |
| `reports/multimodal.json` | Readable machine-readable evaluation counterpart |
| `reports/multimodal-ablation.png` | Chart of all first-run comparisons, including the negative result |

## Cost guardrails

There is no scheduled training, paid CI/GPU job, cloud deployment, premium data feed, or automatic paid fallback. Run the commands locally. Installing packages or downloading public weights must not be confused with buying an inference service. Existing coding-agent use stays within the user's existing access and limits.

## Verification performed

All 30 tests passed both in the original workspace and in a clean export containing only tracked public files. That export contains no publisher PDF/HTML/page images or external encoder weights. Offline neural retraining reproduced every saved model/preprocessing artifact hash. Saved-weight Texas inference matched the original run, and baseline replay completed without network access. See `reports/reproducibility.json`.
