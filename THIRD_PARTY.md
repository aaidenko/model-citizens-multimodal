# Sources and attribution

Our original code is under the MIT license in `LICENSE`. This does not relicense publisher documents, third-party datasets, model weights, or dependencies.

- **Quinnipiac University Poll:** original pre-election PDFs and public archive at https://poll.qu.edu/poll-results/. Authorship and original terms remain with the publisher. URLs, publication dates, captures, hashes, reviewed facts, and non-reconstructive image/text embeddings are included; original PDFs, HTML, and page images are omitted from Git.
- **Marist Poll:** New York October 2022 report, https://maristpoll.marist.edu/polls/the-2022-elections-in-new-york/. The definite-voter Senate row is used. Original documents and images are omitted; provenance is in the manifest.
- **University of Houston Hobby School:** October 2026 Texas report, https://www.uh.edu/hobby/texas2026/statewide26.pdf. Used for factual source verification and an unlabelled inference example. Original report and rendered images are omitted.
- **MIT Election Data and Science Lab:** https://github.com/MEDSL/constituency-returns and https://github.com/MEDSL/2024-elections-official. Used for historical election outcomes and baseline features. Preserve source citation and dataset terms. The archived 2024 state file marks its records unofficial; do not upgrade that status in reporting.
- **Federal Election Commission:** https://www.fec.gov/documents/4227/federalelections2020.pdf and https://www.fec.gov/documents/5675/federalelections2022.pdf. General-election vote totals, including combined-party candidate totals where appropriate, supply the manually curated labels. Hashes are in the label-source manifests.
- **CLIP:** model card https://huggingface.co/openai/clip-vit-base-patch32; original research/code https://github.com/openai/CLIP and its MIT license https://github.com/openai/CLIP/blob/main/LICENSE. The encoder is downloaded separately from a pinned revision. No encoder weights are bundled or sold. Our forecasting head is a separate small learned model.
- **On Point Politics / TPSI public polling feeds:** https://www.tpsielections.com/polls/senate. Discovery source for the statistical research snapshot; numerical observations do not replace primary-source verification. Preserve underlying pollster attribution.
- **Kalshi public market data:** https://docs.kalshi.com/getting_started/quick_start_market_data. Quotes are research proxies, not the official contest benchmark. No trades or authenticated account operations are performed.
- **Organizer:** Senate template and competition requirements are referenced in the README. No endorsement is implied. The user-supplied competition brief is not republished.

PyTorch, Transformers, scikit-learn, NumPy, SciPy, pandas, Pillow, pdfplumber, Requests, Beautiful Soup, Safetensors, and Hugging Face Hub retain their respective licenses. Dependency pins are in the two requirements files. Optional plotting uses Matplotlib.
