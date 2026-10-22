"""Versioned original-document embeddings and matched forecasting ablations."""
from datetime import date
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from citizens.image_inputs import source_views,encoder_pixels
from citizens.multimodal import ENCODER_ID,ENCODER_REVISION
from citizens.v2_data import dump,sha
from citizens.v2_model import neural,scaler,rmse,metrics,predictions,SEEDS
VARIANTS=('numeric','numeric_text','numeric_image','multimodal')


def read_cases(root):
    cases=json.loads((Path(root)/'data/v2/cases.json').read_text())
    seen=set()
    for r in cases:
        if r['id'] in seen:raise ValueError('Duplicate document case')
        seen.add(r['id'])
        if not r.get('visual_review_passed'):raise ValueError('Unreviewed document')
        if r['year']==2024 and r['race'] in {'2024-MI','2024-PA','2024-WI'}:raise ValueError('Consumed test race')
        if (date.fromisoformat(r['election_date'])-date.fromisoformat(r['published_date'])).days<14:raise ValueError('Late document')
        if 'actual_dem_margin' in r:raise ValueError('Labels belong in separate file')
    return cases


def prepare(root,encoder_path):
    import pdfplumber
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    root=Path(root);base=root/'data/v2';cases=read_cases(root)
    model=CLIPModel.from_pretrained(str(encoder_path),local_files_only=True).eval();model.requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(str(encoder_path),local_files_only=True)
    torch.set_num_threads(3);images=[];texts=[];records=[]
    for i,r in enumerate(cases):
        html=r.get('source_kind')=='publisher_html_screenshot'
        if html:
            if sha(root/r['html_path'])!=r['html_sha256'] or sha(root/r['image_path'])!=r['image_sha256'] or sha(root/r['text_path'])!=r['text_sha256']:raise ValueError('Changed HTML capture needs review')
            with Image.open(root/r['image_path']) as im:image=im.convert('RGB').crop(r['crop_pixels'])
            text=(root/r['text_path']).read_text();source_hash=r['html_sha256']
        else:
            path=root/r['pdf_path']
            if sha(path)!=r['pdf_sha256']:raise ValueError('Changed source needs review: '+r['id'])
            with pdfplumber.open(path) as pdf:
                crop=pdf.pages[r['page']-1].crop(r['crop']);image=crop.to_image(resolution=150).original.convert('RGB');text=crop.extract_text() or ''
            source_hash=r['pdf_sha256']
        views,boxes=source_views(image);pixels=encoder_pixels(processor,views)
        words=text.split();chunks=[' '.join(words[j:j+32]) for j in range(0,len(words),32)] or ['']
        tokenized=processor(text=chunks,return_tensors='pt',padding=True,truncation=True,max_length=77)
        with torch.inference_mode():
            vi=model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
            vi=vi/vi.norm(dim=-1,keepdim=True);vector=vi.mean(0);vector=vector/vector.norm()
            tx=model.text_projection(model.text_model(**tokenized).pooler_output)
            tx=tx/tx.norm(dim=-1,keepdim=True);tv=tx.mean(0);tv=tv/tv.norm()
        images.append(vector.numpy());texts.append(tv.numpy())
        # Render inverse-normalized *actual encoder tensors*, not merely source
        # crops. Originals/encoder previews remain excluded from public Git.
        preview=base/'pages'/('encoder-'+r['id']+'.png');preview.parent.mkdir(parents=True,exist_ok=True)
        mean=np.array(processor.image_processor.image_mean);std=np.array(processor.image_processor.image_std)
        arrays=np.clip((pixels.numpy().transpose(0,2,3,1)*std+mean)*255,0,255).round().astype('uint8')
        sheet=Image.new('RGB',(224*min(4,len(views)),224*((len(views)+3)//4)),'#dddddd')
        for j,a in enumerate(arrays):sheet.paste(Image.fromarray(a),((j%4)*224,(j//4)*224))
        sheet.save(preview)
        records.append(dict(id=r['id'],source_sha256=source_hash,source_kind=r.get('source_kind','publisher_pdf'),
            page=r.get('page'),crop=r.get('crop',r.get('crop_pixels')),
            source_pixels=list(image.size),source_boxes=[list(b) for b in boxes],views=len(views),
            first_view_retained_area_fraction=1.0,additional_center_crop=False,text_chunks=len(chunks),
            preview_sha256=sha(preview)))
        print('Encoded',i+1,'/',len(cases),r['id'],flush=True)
    path=base/'embeddings.npz';np.savez_compressed(path,ids=np.array([r['id'] for r in cases]),image=np.array(images),text=np.array(texts))
    dump(base/'embeddings-manifest.json',dict(encoder=ENCODER_ID,revision=ENCODER_REVISION,frozen=True,
        cases_sha256=sha(base/'cases.json'),embeddings_sha256=sha(path),records=records,new_service_spend_usd=0))


def doc_design(cases,emb,train,baselines):
    values=[]
    for r,b in zip(cases,baselines):
        total=r['dem_poll']+r['rep_poll'];margin=100*(r['dem_poll']-r['rep_poll'])/total
        age=(date.fromisoformat(r['election_date'])-date.fromisoformat(r['published_date'])).days
        values.append([(margin-b)/10,(100-total)/10,age/90,np.log(r.get('sample_size') or 600),
            float(r.get('sample_size') is None),float(r.get('population')=='lv'),float(r.get('population') is None)])
    nums=np.array(values);scaled,mean,scale=scaler(nums,train);parts=[scaled];saved=dict(numeric_mean=mean,numeric_scale=scale)
    for name in ('text','image'):
        count=min(4,len(train)-1,emb[name].shape[1]);pca=PCA(n_components=count,svd_solver='full').fit(emb[name][train])
        projected=pca.transform(emb[name]);std=np.maximum(projected[train].std(0),1e-6)
        parts.append(np.clip(projected/std,-5,5).astype(np.float32))
        saved[name+'_mean']=pca.mean_;saved[name+'_components']=pca.components_;saved[name+'_scale']=std
    return parts,saved


def selected_rows(cases,indices):
    # Keep the latest reviewed release per race. Multiple documents do not add
    # observations or inflate reported N. Use stable publisher/ID tie breaking.
    keep={}
    for i in sorted(indices,key=lambda i:(cases[i]['published_date'],cases[i]['publisher'],cases[i]['id'])):keep[cases[i]['race']]=i
    return sorted(keep.values())


def train_documents(root):
    from safetensors.torch import save_file
    root=Path(root);base=root/'data/v2';out=root/'models/v2';cases=read_cases(root)
    manifest=json.loads((base/'embeddings-manifest.json').read_text())
    if sha(base/'cases.json')!=manifest['cases_sha256'] or sha(base/'embeddings.npz')!=manifest['embeddings_sha256']:raise ValueError('Changed embeddings/cases')
    with np.load(base/'embeddings.npz',allow_pickle=False) as b:
        if list(b['ids'])!=[r['id'] for r in cases]:raise ValueError('Embedding order mismatch')
        emb={name:b[name] for name in ('text','image')}
    numeric_run=json.loads((out/'numeric-run.json').read_text());data_manifest=json.loads((base/'manifest.json').read_text())
    if sha(base/'labels.json')!=data_manifest['files']['labels.json']:raise ValueError('Changed outcome labels')
    if sha(base/'numeric-predictions.json')!=numeric_run['numeric_predictions_sha256']:raise ValueError('Changed numeric baseline predictions')
    labels=json.loads((base/'labels.json').read_text());num={r['race']:r for r in json.loads((base/'numeric-predictions.json').read_text())}
    baseline=np.array([num[r['race']]['forward_prediction'] for r in cases]);y=np.array([labels[r['race']]['margin'] for r in cases])
    cal=json.loads((out/'calibration.json').read_text())
    eligible=[i for i,r in enumerate(cases) if not r.get('layout_holdout')]
    train=selected_rows(cases,[i for i in eligible if cases[i]['year']<=2018]);dev=selected_rows(cases,[i for i in eligible if cases[i]['year']==2020])
    final=selected_rows(cases,train+dev);test=selected_rows(cases,[i for i,r in enumerate(cases) if r['year']==2024])
    layout=selected_rows(cases,[i for i,r in enumerate(cases) if r['year']==2024 and r.get('layout_holdout')])
    seen_publishers={cases[i]['publisher'] for i in final}
    unseen=selected_rows(cases,[i for i,r in enumerate(cases) if r['year']==2024 and r['publisher'] not in seen_publishers])
    if min(len(train),len(dev),len(test))<1:raise ValueError('Missing document train/development/test partition')
    parts,pre=doc_design(cases,emb,train,baseline);final_parts,final_pre=doc_design(cases,emb,final,baseline)
    np.savez_compressed(out/'document-preprocessing.npz',**final_pre)
    report=dict(status='EXPERIMENTAL — NOT PROMOTED',new_service_spend_usd=0,encoder=manifest['encoder'],
        revision=manifest['revision'],training_races=len(train),development_races=len(dev),final_training_races=len(final),
        test_races=len(test),publisher_holdout_races=len(layout),unseen_publisher_diagnostic_races=len(unseen),calibration=cal,
        calibration_note='Shared calibrator fitted only to numeric 2022 errors, before new test scoring.',
        baseline=dict(test=metrics(baseline,y,test,cal),publisher_holdout=metrics(baseline,y,layout,cal),
            unseen_publisher_diagnostic=metrics(baseline,y,unseen,cal)),models={},
        cases_sha256=sha(base/'cases.json'),embeddings_sha256=sha(base/'embeddings.npz'),
        numeric_predictions_sha256=sha(base/'numeric-predictions.json'),labels_sha256=sha(base/'labels.json'),
        protocol_sha256=sha(root/'docs/V2_PROTOCOL.md'),device='cpu')
    for variant in VARIANTS:
        columns=[0]+([1] if variant in ('numeric_text','multimodal') else [])+([2] if variant in ('numeric_image','multimodal') else [])
        x=np.concatenate([parts[k] for k in columns],axis=1);xx=np.concatenate([final_parts[k] for k in columns],axis=1)
        grid=[]
        for epoch in (100,300):
            pred,_,_=neural(x,y,baseline,train,epoch,limit=3)
            for weight in (0,.25,.5,1):grid.append(dict(epochs=epoch,weight=weight,development_rmse=rmse(baseline+weight*(pred-baseline),y,dev)))
        chosen=min(grid,key=lambda r:(r['development_rmse'],r['weight'],r['epochs']))
        raw,states,trace=neural(xx,y,baseline,final,chosen['epochs'],limit=3)
        pred=baseline+chosen['weight']*(raw-baseline)
        for seed,state in zip(SEEDS,states):save_file(state,str(out/f'document-{variant}-seed{seed}.safetensors'))
        item=dict(selected=chosen,selection_grid=grid,training_trace=trace,
            trainable_parameters_per_seed=8*xx.shape[1]+17,
            test=metrics(pred,y,test,cal),publisher_holdout=metrics(pred,y,layout,cal),
            unseen_publisher_diagnostic=metrics(pred,y,unseen,cal),
            unblended_test_diagnostic=metrics(raw,y,test,cal),
            test_predictions=predictions(cases,pred,y,test,cal),
            publisher_holdout_predictions=predictions(cases,pred,y,layout,cal))
        if variant=='multimodal':
            import torch
            from citizens.v2_model import network
            changes={};offset=final_parts[0].shape[1]
            for name,k in [('text',1),('image',2)]:
                masked=xx.copy();start=offset if name=='text' else offset+final_parts[1].shape[1]
                masked[:,start:start+final_parts[k].shape[1]]=0;pp=[]
                for state in states:
                    model=network(xx.shape[1],3).eval();model.load_state_dict(state)
                    with torch.inference_mode():pp.append(model(torch.tensor(masked,dtype=torch.float32),torch.tensor(baseline/10,dtype=torch.float32)).numpy()*10)
                changes[name+'_masked_unblended_mean_abs_margin_change']=float(np.mean(np.abs(raw[test]-np.mean(pp,axis=0)[test])))
            item['modality_sensitivity']=changes
        report['models'][variant]=item
    numeric=report['models']['numeric']['test'];mm=report['models']['multimodal']['test']
    report['paired_comparison']=dict(multimodal_minus_numeric_brier=mm['brier']-numeric['brier'],
        multimodal_minus_numeric_rmse=mm['margin_rmse']-numeric['margin_rmse'])
    report['promotion']=dict(promoted=False,reasons=['Small reviewed-document holdout and substantial publisher/layout shift.',
        'A single retrospective cycle cannot establish reliable future skill or contest-winning odds.',
        'Current roster, official benchmark and full prospective input audit remain outstanding.'])
    report['artifacts']={p.name:sha(p) for p in sorted(out.iterdir()) if p.name.startswith('document-') and p.suffix in ('.npz','.safetensors')}
    dump(out/'document-run.json',report);dump(root/'reports/v2-documents.json',report)
    print(json.dumps({name:{'selected':r['selected'],'test':r['test']} for name,r in report['models'].items()},indent=2))
    return report
