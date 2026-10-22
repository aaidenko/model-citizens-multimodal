"""Small CPU-trained forecasting heads with temporal selection and calibration."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
from sklearn.linear_model import Ridge

from citizens.v2_data import dump, sha
SEEDS=(17,29,43)


def scaler(x,indices):
    mean=x[indices].mean(0);scale=x[indices].std(0);constant=scale<1e-6
    scale=np.where(constant,0,scale)
    values=np.clip((x-mean)/np.where(constant,1,scale),-5,5)
    # No train-time variation means no learnable effect. Novel holdout values
    # must not activate arbitrary, untrained neural weights on that column.
    values[:,constant]=0
    return values.astype(np.float32),mean,scale


def network(dim,limit):
    import torch
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(dim,8),torch.nn.Tanh(),torch.nn.Linear(8,1))
        def forward(self,x,base):return base+(limit/10)*torch.tanh(self.layers(x).squeeze(-1))
    return Head()


def neural(x,y,base,indices,epoch,limit=6):
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    xx=torch.tensor(x,dtype=torch.float32);bb=torch.tensor(base/10,dtype=torch.float32)
    yy=torch.tensor(y/10,dtype=torch.float32);preds=[];states=[];traces=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=network(x.shape[1],limit)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.05)
        first=None
        for step in range(epoch):
            pred=model(xx[indices],bb[indices]);loss=(pred-yy[indices]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if step==0:first=float(loss.detach())*100
        model.eval()
        with torch.inference_mode():preds.append(model(xx,bb).numpy()*10)
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        traces.append(dict(seed=seed,first_loss=first,last_loss=float(loss.detach())*100,
            parameter_l2_change=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt())))
    return np.mean(preds,axis=0),states,traces


def fitted(x,y,base,indices,config):
    scaled,mean,scale=scaler(x,indices)
    if config['kind']=='ridge':
        fit=Ridge(alpha=config['alpha']).fit(scaled[indices],(y-base)[indices])
        pred=base+np.clip(fit.predict(scaled),-6,6)
        artifact=dict(kind='ridge',alpha=config['alpha'],coef=fit.coef_.tolist(),intercept=float(fit.intercept_))
        states=[];trace=[]
    else:
        pred,states,trace=neural(scaled,y,base,indices,config['epochs'])
        artifact=dict(kind='neural',epochs=config['epochs'],limit=6)
    return pred,dict(**artifact,mean=mean.tolist(),scale=scale.tolist()),states,trace


def rmse(pred,y,indices):return float(np.sqrt(np.mean((pred[indices]-y[indices])**2)))


def fit_calibration(pred,y,indices,years):
    if any(years[i]!=2022 for i in indices):raise ValueError('Calibration must use 2022 only')
    error=(y-pred)[indices]
    def objective(params):return float(-t.logpdf((error-params[0])/np.exp(params[1]),df=5).sum()+len(error)*params[1])
    result=minimize(objective,[float(np.median(error)),np.log(max(float(np.std(error)),1))],
        method='L-BFGS-B',bounds=[(-15,15),(np.log(1),np.log(40))])
    if not result.success:raise ValueError('Calibration optimization failed: '+result.message)
    return dict(df=5,location_points=float(result.x[0]),scale_points=float(np.exp(result.x[1])),
                races=len(indices),fit_years=[2022],objective='Student-t margin error maximum likelihood')


def probability(margin,calibration):
    return t.cdf((np.asarray(margin)+calibration['location_points'])/calibration['scale_points'],df=calibration['df'])


def metrics(pred,y,indices,calibration):
    if not len(indices):return dict(races=0)
    q=probability(pred[indices],calibration);truth=(y[indices]>0).astype(float)
    bins=[]
    for lo,hi in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
        mask=(q>=lo)&(q<hi)
        if mask.any():bins.append(dict(lower=lo,upper=min(hi,1),races=int(mask.sum()),mean_probability=float(q[mask].mean()),democratic_win_fraction=float(truth[mask].mean())))
    return dict(races=len(indices),brier=float(np.mean((q-truth)**2)),
        log_loss=float(-np.mean(truth*np.log(np.clip(q,1e-9,1))+(1-truth)*np.log(np.clip(1-q,1e-9,1)))),
        accuracy=float(np.mean((q>=.5)==truth)),margin_rmse=rmse(pred,y,indices),reliability=bins)


def predictions(rows,pred,y,indices,cal):
    return [dict(race=rows[i]['race'],year=rows[i]['year'],predicted_dem_margin=float(pred[i]),
        actual_dem_margin=float(y[i]),q_democrat=float(probability(pred[i],cal))) for i in indices]


def train_numeric(root):
    from safetensors.torch import save_file
    start=time.monotonic();root=Path(root);base=root/'data/v2';out=root/'models/v2';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((base/'snapshots.json').read_text());labels=json.loads((base/'labels.json').read_text())
    manifest=json.loads((base/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if sha(base/name)!=h:raise ValueError('Dataset changed: '+name)
    x=np.asarray([r['x'] for r in rows]);y=np.asarray([labels[r['race']]['margin'] for r in rows]);poll=x[:,0]
    years=[r['year'] for r in rows]
    tr=[i for i,r in enumerate(rows) if r['split']=='train'];dev=[i for i,r in enumerate(rows) if r['split']=='development']
    cal=[i for i,r in enumerate(rows) if r['split']=='calibration'];test=[i for i,r in enumerate(rows) if r['split']=='test']
    configs=[dict(kind='ridge',alpha=a) for a in (4,40,400)]+[dict(kind='neural',epochs=e) for e in (100,300)]
    grid=[]
    for config in configs:
        pred,_,_,_=fitted(x,y,poll,tr,config)
        grid.append(dict(config=config,development_rmse=rmse(pred,y,dev)))
    selected=min(grid,key=lambda r:r['development_rmse'])['config']
    dev_pred,_,_,_=fitted(x,y,poll,tr,selected)
    pred,artifact,states,trace=fitted(x,y,poll,tr+dev,selected)
    # Always retain a genuinely trained neural comparator, even if a regularized
    # ridge model wins selection. This is not a post-test model choice.
    neural_config=min([g for g in grid if g['config']['kind']=='neural'],key=lambda r:r['development_rmse'])['config']
    neural_pred,neural_artifact,neural_states,neural_trace=fitted(x,y,poll,tr+dev,neural_config)
    for seed,state in zip(SEEDS,neural_states):save_file(state,str(out/f'numeric-seed{seed}.safetensors'))
    dump(out/'numeric.json',artifact);dump(out/'numeric-neural.json',neural_artifact)
    calibration=fit_calibration(pred,y,cal,years);dump(out/'calibration.json',calibration)
    crossfit=np.array(poll,copy=True)
    for year in sorted(set(years)):
        if year>2020:continue
        before=[i for i,r in enumerate(rows) if r['year']<year]
        current=[i for i,r in enumerate(rows) if r['year']==year]
        if len(before)>=20:
            p,_,_,_=fitted(x,y,poll,before,selected);crossfit[current]=p[current]
    # 2022/2024 predictions use the model frozen after 2020; earlier rows receive
    # forward predictions for training the document residual head.
    crossfit[np.array(years)>2020]=pred[np.array(years)>2020]
    dump(base/'numeric-predictions.json',[dict(race=r['race'],year=r['year'],
        prediction=float(pred[i]),forward_prediction=float(crossfit[i])) for i,r in enumerate(rows)])
    errors=y[dev]-dev_pred[dev];cycle_means=[float(errors[np.array(years)[dev]==year].mean()) for year in (2018,2020)]
    total=float(np.mean(errors**2));between=float(np.var(cycle_means))
    report=dict(status='RETROSPECTIVE RESEARCH — NOT PROMOTED',new_service_spend_usd=0,
        data=manifest,protocol_sha256=sha(root/'docs/V2_PROTOCOL.md'),selection=grid,selected=selected,
        selected_neural=neural_config,neural_training_trace=neural_trace,
        final_fit_last_year=2020,calibration=calibration,
        development=dict(races=len(dev),selected_margin_rmse=rmse(dev_pred,y,dev),raw_poll_margin_rmse=rmse(poll,y,dev)),
        calibration_fit_diagnostic=metrics(pred,y,cal,calibration),
        test=dict(selected=metrics(pred,y,test,calibration),raw_poll_same_calibration=metrics(poll,y,test,calibration),
                  neural_same_calibration=metrics(neural_pred,y,test,calibration)),
        test_predictions=predictions(rows,pred,y,test,calibration),
        calibration_predictions=predictions(rows,pred,y,cal,calibration),
        cycle_error_diagnostic=dict(development_cycle_means=cycle_means,total_mean_square=total,
            between_cycle_variance=between,rough_latent_error_correlation=min(.8,between/max(total,1e-9)),
            warning='Only two development election cycles; descriptive, not a reliably estimated correlation.'),
        runtime_seconds=round(time.monotonic()-start,3),device='cpu',
        numeric_predictions_sha256=sha(base/'numeric-predictions.json'))
    report['artifacts']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and
        (p.name=='calibration.json' or (p.name.startswith('numeric') and p.name!='numeric-run.json'))}
    dump(root/'reports/v2-numeric.json',report);dump(out/'numeric-run.json',report)
    print(json.dumps({'selected':selected,'test':report['test'],'calibration':calibration},indent=2),flush=True)
    return report
