"""Build versioned numeric snapshots offline; optional source directory seeds cache."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from citizens.v2_data import build

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path);args=p.parse_args()
    print(json.dumps(build(ROOT,args.source_dir),indent=2))
