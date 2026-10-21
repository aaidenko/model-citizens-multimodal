#!/usr/bin/env python3
"""Prepare a local public-source PDF for visual review, without any model API."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pdfplumber


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf',type=Path)
    p.add_argument('--source-url',required=True)
    p.add_argument('--slug',required=True)
    p.add_argument('--pages',required=True,help='One-based PDF page numbers, comma separated')
    args=p.parse_args()
    if not re.fullmatch('[a-z0-9-]+',args.slug):
        p.error('Use a lowercase, hyphenated slug')
    pages=sorted({int(v) for v in args.pages.split(',')})
    if not pages or min(pages)<1 or len(pages)>8:
        p.error('Select between one and eight pages')
    root=Path(__file__).resolve().parent
    out=root/'data/evidence'/args.slug
    out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    result=dict(source_url=args.source_url,sha256=digest,visual_review_passed=False,
                extracted_observations_approved=False,pages=[])
    renderer=shutil.which('pdftoppm')
    with pdfplumber.open(args.pdf) as pdf:
        if max(pages)>len(pdf.pages):
            p.error('Selected page is outside the document')
        for n in pages:
            page=pdf.pages[n-1]
            text=page.extract_text() or ''
            result['pages'].append(dict(page_1_based=n,text=text,tables=page.extract_tables(),
                needs_ocr_or_visual_extraction=not bool(text.strip())))
            if renderer:
                subprocess.run([renderer,'-f',str(n),'-l',str(n),'-scale-to','1400',
                                '-png',str(args.pdf),str(out/'page')],check=True)
    result['rendered']=bool(renderer)
    result['next_action']='Inspect page images, verify the numbers and method, record reviewer and exact source hash, then approve a structured observation. Extraction alone is not verification.'
    (out/'review.json').write_text(json.dumps(result,indent=2))
    print(out/'review.json')


if __name__=='__main__':
    main()
