"""Inference from saved local weights; no fitting or network requests."""
import json
from pathlib import Path
import numpy as np
from citizens.v2_data import sha
from citizens.v2_documents import read_cases
from citizens.v2_model import network,probability,SEEDS


def normalize(values,mean,scale):
    values=np.asarray(values);mean=np.asarray(mean);scale=np.asarray(scale)
    out=np.clip((values-mean)/np.where(scale==0,1,scale),-5,5)
    out[...,scale==0]=0
    return out.astype(np.float32)


def predict(root,case_id,variant='multimodal'):
    import torch
    from safetensors.torch import load_file
    from datetime import date
    root=Path(root);base=root/'data/v2';out=root/'models/v2'
    run=json.loads((out/'document-run.json').read_text());cases=read_cases(root)
    if sha(base/'cases.json')!=run['cases_sha256'] or sha(base/'embeddings.npz')!=run['embeddings_sha256']:raise ValueError('Dataset differs from trained run')
    if sha(base/'numeric-predictions.json')!=run['numeric_predictions_sha256']:raise ValueError('Changed numeric baseline inputs')
    for name,h in run['artifacts'].items():
        if sha(out/name)!=h:raise ValueError('Changed trained artifact: '+name)
    ids=[r['id'] for r in cases]
    if case_id not in ids or variant not in run['models']:raise ValueError('Unknown case or variant')
    i=ids.index(case_id);r=cases[i]
    stored={r['race']:r for r in json.loads((base/'numeric-predictions.json').read_text())}
    baseline=stored[r['race']]['forward_prediction']
    total=r['dem_poll']+r['rep_poll'];margin=100*(r['dem_poll']-r['rep_poll'])/total
    age=(date.fromisoformat(r['election_date'])-date.fromisoformat(r['published_date'])).days
    nums=[(margin-baseline)/10,(100-total)/10,age/90,np.log(r.get('sample_size') or 600),
          float(r.get('sample_size') is None),float(r.get('population')=='lv'),float(r.get('population') is None)]
    with np.load(out/'document-preprocessing.npz',allow_pickle=False) as p,np.load(base/'embeddings.npz',allow_pickle=False) as emb:
        if list(emb['ids'])!=ids:raise ValueError('Embedding order mismatch')
        parts=[normalize(nums,p['numeric_mean'],p['numeric_scale'])]
        for name in ('text','image'):
            v=(emb[name][i]-p[name+'_mean'])@p[name+'_components'].T
            parts.append(np.clip(v/p[name+'_scale'],-5,5).astype(np.float32))
    cols=[0]+([1] if variant in ('numeric_text','multimodal') else [])+([2] if variant in ('numeric_image','multimodal') else [])
    x=torch.tensor(np.concatenate([parts[j] for j in cols])[None,:],dtype=torch.float32)
    model=network(x.shape[1],3).eval();values=[]
    for seed in SEEDS:
        model.load_state_dict(load_file(str(out/f'document-{variant}-seed{seed}.safetensors')))
        with torch.inference_mode():values.append(float(model(x,torch.tensor([baseline/10],dtype=torch.float32))[0])*10)
    raw=float(np.mean(values));weight=run['models'][variant]['selected']['weight'];pred=baseline+weight*(raw-baseline)
    return dict(case_id=case_id,variant=variant,status=run['status'],predicted_dem_margin=pred,
        q_democrat=float(probability(pred,run['calibration'])),document_weight=weight,
        unblended_neural_margin_diagnostic=raw,base_numeric_margin=baseline,new_service_spend_usd=0,
        note='Historical training fit' if r['year']<=2020 else 'Frozen-model historical inference')
