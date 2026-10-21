"""Train/reproduce the $0 local multimodal forecasting experiment."""
import argparse
from pathlib import Path
from citizens.multimodal import prepare, train

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encode', action='store_true', help='Recreate embeddings from original reviewed PDFs')
    ap.add_argument('--encoder-path', type=Path, help='Local pinned CLIP directory, required for --encode')
    args = ap.parse_args()
    if args.encode:
        if args.encoder_path is None:
            ap.error('--encode requires --encoder-path')
        prepare(ROOT, args.encoder_path)
    train(ROOT)
