import argparse
import json
from pathlib import Path
from citizens.v2_inference import predict
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case_id');p.add_argument('--variant',default='multimodal',choices=['numeric','numeric_text','numeric_image','multimodal']);args=p.parse_args()
    print(json.dumps(predict(Path(__file__).resolve().parent,args.case_id,args.variant),indent=2))
