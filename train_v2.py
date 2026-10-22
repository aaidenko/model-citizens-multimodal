"""Train free local v0.2 models from included numeric snapshots/embeddings."""
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--documents',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent
    if args.documents:
        from citizens.v2_documents import train_documents
        train_documents(root)
    else:
        from citizens.v2_model import train_numeric
        train_numeric(root)
