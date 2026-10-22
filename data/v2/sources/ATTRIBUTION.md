# Source data attribution

The FiveThirtyEight CSV captures in this directory and their derived factual tables retain **CC BY 4.0** attribution. See `LICENSE-538.txt`; the repository's MIT code license does not replace that license.

- Historical data: https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/raw_polls.csv
- Codebook: https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/README.md
- Original license: https://github.com/fivethirtyeight/data/blob/master/LICENSE
- Archived 2024 capture: https://web.archive.org/web/20241129175751id_/https://projects.fivethirtyeight.com/polls-page/data/senate_polls.csv

The adjacent source JSON files retain retrieval metadata and hashes. Numeric source CSV bytes are preserved unchanged. Derived data selects eligible D/R general-election races, enforces dated cutoffs, deduplicates questions/pollsters, aggregates source facts, normalizes two-party margins, and separates targets from features. Actual election results and retrospective ratings in original captures must not enter the forecasting feature whitelist. Strictly previous presidential outcomes and previous Senate winners may supply dated historical features.

MEDSL returns are cited separately in the repository README and `THIRD_PARTY.md`. Publisher originals, HTML, screenshots and large pretrained weights are excluded from public Git; their original authorship and terms remain with the sources. Derived embeddings and our small forecasting checkpoints are included for offline reproduction, not as a replacement for the original documents.
