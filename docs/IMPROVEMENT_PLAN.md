# Improving the multimodal forecast

Recommendations recorded October 21, 2026. These are proposed experiments, not measured accuracy gains. The original experiment, features, weights, and results remain unchanged.

## What the diagnostic found

**Preprocessing defect:** v0.1 inherited CLIP's default square center crop after resizing the shortest edge to 224 pixels. Wide poll tables consequently lose much of their content. The 2016 Colorado source crop is about 612 × 137 PDF points; the model keeps only the central horizontal span, approximately x=237–375. Both candidate-name table cells at x≈36 and the 56/38 topline cells at x≈162 fall outside it. The retained area is approximately 22.4%.

This is a flaw in the initial implementation. It gives a concrete first repair, but does not establish how much forecast accuracy will improve after repair. `scripts/audit_image_inputs.py` records the geometry diagnostic in `reports/image-input-audit.json` without changing the model. The pinned processor and the [CLIP preprocessing documentation](https://huggingface.co/docs/transformers/en/model_doc/clip) describe the crop behavior.

**Data limitations:** the training set has only 20 independent race outcomes, including five Democratic and fifteen Republican wins. Its average actual-minus-poll two-party margin is about −4.37 points. This is a property of this convenience sample, not proof that winner imbalance causes a particular model bias. The two validation races are both comfortable Democratic wins. The three test races are all close contests from the same cycle. These sets are too small and narrow to select a reliable correction or calibrate win probabilities.

**Representation limits:** image and text inputs describe the same poll already represented numerically. A second representation does not create an independent source. The model currently receives no numeric sample-size or pollster-history feature, and its likely-voter flag is constant. A hard-coded six-point error scale turns margins into uncalibrated probabilities.

## Recommended order

1. **Repair image visibility.** Preserve the complete source crop with aspect-ratio-preserving padding, and add separate readable tiles for the question, candidate names, and topline/subgroup tables. Padding an entire wide page alone makes tiny text smaller; tiling is needed as well. Verify the actual processed image sent to the encoder, not just the original source page. Retain v0.1 unchanged; create versioned v0.2 embeddings. First evaluate text/number visibility and extraction accuracy, then forecasting performance.

2. **Expand independent outcomes and source diversity.** Aim first for 100 or more usable race–election outcomes from several cycles and publishers, subject to free archive coverage. That is a collection target, not a sufficient-sample guarantee. Add both parties' wins and close/safe contests through a predefined coverage rule. Do not artificially balance winners 50/50 and treat that as a representative forecasting population. Multiple documents from one race can supply a richer snapshot but remain one shared election outcome.

3. **Give the image/text branch useful evidence to read.** Extract and verify sample size, voter population, field/publication dates, poll mode, undecided share, named matchups, and subgroup sample sizes where reported. Keep missing values explicit. Use native PDF text/table extraction first; a free local OCR model can handle image-only material. The strongest testable document benefit may be recovering genuine polls absent from the structured feed, or improving uncertainty estimates. Cross-race comparisons within one poll are another hypothesis, not independent samples or an automatic adjustment.

4. **Train a conservative forecasting combination.** Use a stronger numeric baseline: multiple recent pollsters, recency, prior state lean, incumbency, and a dated national environment where historical sources support them. Train the multimodal branch to add a regularized correction or estimate uncertainty around that baseline. Learn its contribution from earlier-cycle validation and allow the contribution to be zero. Keep encoders frozen initially; train only the small forecasting head. Larger heads or encoder fine-tuning come after a larger dataset and evidence of a useful signal.

5. **Fit uncertainty and assess probability quality.** Replace the fixed six-point conversion with a low-complexity uncertainty/calibration model fitted to predictions for races the model did not train on. Account for shared election-cycle/pollster errors instead of assuming every poll is independent. Track Brier score, log loss, reliability, and winner accuracy. The competition's reward is unusual, so choose contest picks from calibrated probabilities using the separate scoring policy. Winner accuracy alone is not the competition objective. [Scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html) explains probability scoring and calibration; no package upgrade is required for this recommendation.

6. **Use a locked evaluation plan.** Keep every poll/page from a race in the same partition, train on earlier cycles, and compare numeric-only, numeric+text, numeric+image, and all inputs on the same examples. Fit scalers and projections inside each training partition. Include an additional publisher/layout holdout. The three v0.1 2024 races are consumed development evidence; do not tune until they turn into wins. Preselect new unexamined cases before scoring, disclose any cycle-level information already used, and retain 2026 prospective results when they become available. Any new encoder used for a claimed historical forecast must also have an appropriate pre-election release/training history.

## Suggested architecture

Original PDFs/charts → image-and-text document reader → verified observations and document features → small trained forecasting network alongside the numeric baseline → calibrated probabilities → contest pick selection.

An optional direct image/text feature branch can remain, but must beat the same system without that branch. If extracted numeric evidence explains all of its benefit, report that honestly. The system remains multimodal and uses a trained AI forecaster; it does not need to infer election outcomes from document appearance.

## Free implementation options

The first changes can use the existing Pillow, pdfplumber, scikit-learn, and PyTorch environment on the local CPU. [Tesseract](https://github.com/tesseract-ocr/tesseract) is an open-source local OCR option for scans. The [DePlot research](https://arxiv.org/abs/2212.10505) supports testing a chart-to-table approach as a document-reading architecture, not a claim about election forecasting. A separate chart-reader benchmark should establish exact numeric extraction before any new model is adopted. Newer vision-language models require checks of license, local resource needs, and historical leakage; downloading a bigger model is not itself evidence of an improvement.

No paid API, cloud GPU, subscription, synthetic election labels, or new account is needed for the proposed initial work. Do not claim a measured accuracy lift until the locked evaluation supports it.
