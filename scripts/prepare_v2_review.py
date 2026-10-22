"""Prepare original crops for inspection. This script cannot approve reviews."""
from datetime import date
import json
from pathlib import Path
import re
import sys

import pdfplumber

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.v2_data import dump,surname,sha
BASE=ROOT/'data/v2';PAGES=BASE/'pages';PAGES.mkdir(parents=True,exist_ok=True)


def main():
    candidates=json.loads((BASE/'document-candidates.json').read_text())
    snapshots={r['race']:r for r in json.loads((BASE/'snapshots.json').read_text())}
    old={r['id']:r for r in json.loads((ROOT/'data/multimodal/cases.json').read_text())}
    output=[];errors=[]
    for i,c in enumerate(candidates):
        if c.get('error') or snapshots[c['race']]['split']=='consumed':continue
        if c.get('v1_review'):
            r=old[c['v1_review']]
            # Inherit only the exact previously inspected bytes and crop.
            if sha(ROOT/c['path'])!=r['pdf_sha256']:raise ValueError('Changed v0.1 PDF')
            case={**r,'id':'v2-'+r['id'],'pdf_path':c['path'],'publisher':c['publisher'],
                'sample_size':None,'population':'lv' if r.get('likely_voters') else None,
                'review_origin':r['id'],'layout_holdout':False}
            case.pop('actual_dem_margin',None)
            output.append(case);continue
        try:
            matches=[]
            with pdfplumber.open(ROOT/c['path']) as pdf:
                for pno,page in enumerate(pdf.pages[:40]):
                    words=page.extract_words();text=' '.join(w['text'] for w in words);lower=text.lower()
                    if surname(c['dem_candidate']) not in lower or surname(c['rep_candidate']) not in lower:continue
                    names=[w for w in words if surname(c['dem_candidate']) in w['text'].lower() or surname(c['rep_candidate']) in w['text'].lower()]
                    numeric_tokens=set(re.findall(r'\b\d+(?:\.\d+)?',text))
                    values=sum(str(c[k]).removesuffix('.0') in numeric_tokens for k in ('dem_poll','rep_poll'))
                    # Prefer the Senate table over narrative mentions; selection
                    # uses source text and feed values, never outcome labels.
                    score=values*10+min(len(names),6)+(3 if 'senat' in lower else 0)
                    matches.append((score,pno,words,names))
                if not matches:raise ValueError('No candidate-pair page found')
                _,pno,words,names=max(matches,key=lambda r:(r[0],-r[1]));page=pdf.pages[pno]
                top=max(0,min(w['top'] for w in names)-65);bottom=min(page.height,max(w['bottom'] for w in names)+100)
                crop=[0,round(top,2),float(page.width),round(bottom,2)]
                image=page.crop(crop).to_image(resolution=130).original.convert('RGB')
                cid=c['race']+'-'+re.sub(r'[^a-z0-9]+','-',c['publisher'].lower()).strip('-')
                path=PAGES/(cid+'.png');image.save(path)
                output.append(dict(id=cid,race=c['race'],year=int(c['race'][:4]),state=c['race'][-2:],
                    election_date='2024-11-05',published_date=c['published_date'],publisher=c['publisher'],
                    pdf_path=c['path'],pdf_url=c['url'],pdf_sha256=c['sha256'],page=pno+1,crop=crop,
                    dem_candidate=c['dem_candidate'],rep_candidate=c['rep_candidate'],dem_poll=c['dem_poll'],rep_poll=c['rep_poll'],
                    sample_size=c.get('sample_size'),population=c.get('population'),
                    image_path=str(path.relative_to(ROOT)),visual_review_passed=False,
                    layout_holdout=any(s in c['publisher'].lower() for s in ('monmouth','emerson'))))
        except Exception as exc:errors.append(dict(url=c['url'],race=c['race'],error=str(exc)[:160]))
        if (i+1)%10==0:print('Prepared',i+1,'/',len(candidates),flush=True)
    dump(BASE/'review-candidates.json',output);dump(BASE/'review-errors.json',errors)
    print(json.dumps(dict(crops=len(output),inherited_reviews=sum(r['visual_review_passed'] for r in output),
        new_pending=sum(not r['visual_review_passed'] for r in output),errors=len(errors)),indent=2))


if __name__=='__main__':main()
