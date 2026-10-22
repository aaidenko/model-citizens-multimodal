"""Limited cell-reading benchmark; OCR output is not a visual review."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pdfplumber
from transformers import CLIPProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.local_ocr import reader,tokens,row_value
from citizens.v2_data import dump,sha,surname


def main():
    p=argparse.ArgumentParser();p.add_argument('--encoder-path',type=Path,required=True);args=p.parse_args()
    processor=CLIPProcessor.from_pretrained(str(args.encoder_path),local_files_only=True)
    engine,models=reader();cases=json.loads((ROOT/'data/v2/cases.json').read_text())
    wanted=['v2-2016-CO-2016-10-18','v2-2014-NJ-2014-10-02','v2-2018-CT-2018-10-10','2024-AZ-yougov','2024-TX-yougov']
    results=[];counts={k:0 for k in ('old_center_crop','preserved_full_source')};pairs={k:0 for k in counts}
    for cid in wanted:
        r=next(r for r in cases if r['id']==cid)
        with pdfplumber.open(ROOT/r['pdf_path']) as pdf:image=pdf.pages[r['page']-1].crop(r['crop']).to_image(resolution=150).original.convert('RGB')
        pixel=processor(images=image,return_tensors='np')['pixel_values'][0].transpose(1,2,0)
        mean=np.array(processor.image_processor.image_mean);std=np.array(processor.image_processor.image_std)
        old=Image.fromarray(np.clip((pixel*std+mean)*255,0,255).round().astype('uint8'))
        item=dict(id=cid,expected=[r['dem_poll'],r['rep_poll']],source_sha256=r['pdf_sha256'],visual_review_passed=False)
        for name,im in [('old_center_crop',old),('preserved_full_source',image)]:
            words=tokens(engine,im);values=[row_value(words,surname(r[k])) for k in ('dem_candidate','rep_candidate')]
            correct=[v is not None and abs(v-g)<.05 for v,g in zip(values,item['expected'])]
            counts[name]+=sum(correct);pairs[name]+=int(all(correct));item[name]=dict(extracted=values,correct_cells=sum(correct))
        results.append(item);print(cid,item['preserved_full_source'],flush=True)
    dump(ROOT/'reports/v2-ocr.json',dict(status='LIMITED LOCAL READER DIAGNOSTIC',cases=len(results),
        correct_cells=counts,total_cells=2*len(results),exact_pairs=pairs,results=results,
        packages={k:metadata.version(k) for k in ('rapidocr','onnxruntime','opencv-python-headless')},
        model_hashes={p.name:sha(p) for p in models.values()},new_service_spend_usd=0,
        limitations=['Five reviewed convenience cases, not a general chart-reading accuracy estimate.',
            'Candidate-as-column layouts are intentionally unsupported and failures are retained.',
            'Compares the old cropped encoder image to preserved full-resolution source for the OCR reader, not to new encoder tiles.',
            'OCR is an optional extraction fallback; these outputs do not enter the forecasting model or automatically approve a source.',
            'The OCR release is newer than the historical elections; this is a present-day extraction diagnostic only.']))


if __name__=='__main__':main()
