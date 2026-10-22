"""One local verification command; optional offline retraining, never installation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retrain',action='store_true');args=p.parse_args()
    commands=[]
    if args.retrain:commands=[['scripts/build_v2_data.py'],['train_v2.py'],['train_v2.py','--documents']]
    commands.extend([['scripts/audit_v2_inputs.py'],['run.py','--as-of','2026-10-21'],['-m','unittest','discover','-s','tests','-v']])
    for command in commands:subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    print('Local verification completed. No sources were refreshed or submission sent.')
