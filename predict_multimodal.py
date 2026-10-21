import argparse
import json
from pathlib import Path
from citizens.multimodal import predict

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run saved multimodal weights on a reviewed encoded case, offline.')
    ap.add_argument('--case', default='2026-TX-2026-10-21')
    args = ap.parse_args()
    print(json.dumps(predict(Path(__file__).resolve().parent, args.case), indent=2))
