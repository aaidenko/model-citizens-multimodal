"""Optional local OCR reader. Its output never grants visual approval."""
from importlib import metadata
from pathlib import Path
import re


def reader():
    import rapidocr
    from rapidocr import RapidOCR
    if metadata.version('rapidocr')!='3.10.0':raise ValueError('Install the pinned optional OCR requirements')
    models=Path(rapidocr.__file__).parent/'models'
    names={'Det':'PP-OCRv6_det_small.onnx','Cls':'ch_ppocr_mobile_v2.0_cls_mobile.onnx','Rec':'PP-OCRv6_rec_small.onnx'}
    paths={k:models/v for k,v in names.items()}
    if not all(p.is_file() for p in paths.values()):raise ValueError('Bundled OCR files missing; no implicit network download allowed')
    params={k+'.model_path':str(v) for k,v in paths.items()}
    params.update({'EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':2})
    return RapidOCR(params=params),paths


def tokens(engine,image):
    result=engine(image);output=[]
    if result.txts is None:return output
    for text,score,box in zip(result.txts,result.scores,result.boxes):
        coords=box.tolist()
        output.append(dict(text=str(text),confidence=float(score),
            left=min(p[0] for p in coords),right=max(p[0] for p in coords),
            top=min(p[1] for p in coords),bottom=max(p[1] for p in coords)))
    return output


def row_value(words,name):
    """Read a simple candidate-row table's first numeric column, or return None.

    This does not handle arbitrary charts or candidate-as-column layouts.
    Ambiguity/missing values require review, never a confident guessed value.
    """
    name=name.lower()
    labels=[w for w in words if re.search(r'\b'+re.escape(name)+r'\b',w['text'].lower()) and len(w['text'])<35]
    for label in sorted(labels,key=lambda w:(w['top'],w['left'])):
        inline=re.search(re.escape(name)+r'\s+(\d+(?:\.\d+)?)\s*%?',label['text'],re.I)
        if inline:return float(inline[1])
        mid=(label['top']+label['bottom'])/2;height=label['bottom']-label['top']
        matches=[]
        for w in words:
            text=w['text'].strip().replace('%','')
            if not re.fullmatch(r'\d+(?:\.\d+)?',text):continue
            if w['left']<label['right']-2 or abs((w['top']+w['bottom'])/2-mid)>max(7,.7*height):continue
            if 0<=float(text)<=100:matches.append(w)
        if matches:return float(min(matches,key=lambda w:w['left'])['text'].strip().replace('%',''))
    return None
