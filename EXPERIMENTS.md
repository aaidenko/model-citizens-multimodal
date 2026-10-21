# Multimodal experiments worth testing

The useful initial multimodal capability is evidence extraction and verification from public poll documents. It creates a possible coverage advantage without a new paid data feed. It does not by itself establish a forecasting advantage.

## 1. Additional poll coverage from charts and PDFs

**Hypothesis:** Primary releases contain usable toplines, candidate names, dates, voter-screen definitions, and methodology that a structured feed omits or misstates.

**Current example:** The University of Houston Texas report's chart and overall table agree, and its methodology resolves a conflicting tracker field-end date. The pipeline replaces the existing occurrence of that poll rather than adding a second one.

**Evaluation:** For past elections, archive what was actually public before a fixed cutoff. Compare the same model with structured feeds alone against the model with verified document-only observations. Evaluate probability calibration and decision changes, not just extraction success. Record extraction errors and missing documents as well as successes.

## 2. Candidate performance relative to another race in the same poll

**Hypothesis:** A report containing several races may help distinguish statewide political conditions from candidate-specific strength.

**Current example:** The same Texas PDF reports a Senate D-minus-R margin of +2 and a governor D-minus-R margin of -3. That 5-point contrast is a potential descriptive feature. It is not two independent samples and is not a validated adjustment.

**Evaluation:** Use paired historic releases, retaining the covariance from shared respondents where available. Test a regularized feature against the topline-only baseline on entire held-out election cycles. The Senate topline already contains candidate information, so the additional feature may add nothing.

## 3. Uncertainty from methodology and missingness

**Hypothesis:** A document's voter-screen definition, reported weighting, missing field dates, subgroup sample sizes, and unusually large undecided share can improve uncertainty estimates or source-quality flags.

**Evaluation:** Use explicit, auditable fields instead of asking a model to assign an unexplained “quality score.” Train the uncertainty relationship on historical data, then test calibration on later cycles. A missing subgroup sample size must remain missing; it cannot be inferred from a statewide total.

## Promotion rule

Keep each experimental feature disabled until its out-of-cycle evaluation improves the chosen metric against the same baseline and survives sensible sensitivity tests. Record failures. Do not use candidate appearance, facial expressions, crowd size, or impressions from debate clips as probability adjustments without a separately justified and validated method.

All extraction and tests can use installed local libraries and visual review within existing agent access. No paid multimodal API is required.
