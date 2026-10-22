import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.v2_documents import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder-path',type=Path,required=True);args=p.parse_args()
    prepare(ROOT,args.encoder_path)
