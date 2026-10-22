"""Explicitly restore exact publisher bytes. Changed bytes are quarantined."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import requests

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true',help='Fetch missing public publisher originals');args=p.parse_args()
    cases=json.loads((ROOT/'data/v2/cases.json').read_text());sources={}
    for r in cases:
        html=r.get('source_kind')=='publisher_html_screenshot'
        url=r['release_url'] if html else r['pdf_url'];path=r['html_path'] if html else r['pdf_path'];digest=r['html_sha256'] if html else r['pdf_sha256']
        sources[url]=(path,digest,html)
    if not args.all:
        print(json.dumps({'sources':len(sources),'message':'Pass --all to fetch missing public originals. No network request was made.'}));raise SystemExit(0)
    for url,(path,expected,html) in sources.items():
        try:
            if urlparse(url).scheme!='https':raise ValueError('Only HTTPS sources permitted')
            destination=(ROOT/path).resolve()
            if not destination.is_relative_to(ROOT/'data'):raise ValueError('Source path outside data directory')
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==expected:continue
            response=requests.get(url,timeout=30);response.raise_for_status();content=response.content
            actual=hashlib.sha256(content).hexdigest();destination.parent.mkdir(parents=True,exist_ok=True)
            if actual!=expected:
                quarantine=ROOT/'data/v2/raw'/('changed-'+actual[:20]);quarantine.parent.mkdir(parents=True,exist_ok=True);quarantine.write_bytes(content)
                print(json.dumps({'url':url,'status':'CHANGED — NOT REVIEWED','sha256':actual}));continue
            if not html and not content.startswith(b'%PDF'):raise ValueError('Not a PDF')
            destination.write_bytes(content);print(json.dumps({'url':url,'status':'Exact source restored'}))
        except Exception as e:print(json.dumps({'url':url,'status':'Unavailable','error':str(e)[:150]}))
