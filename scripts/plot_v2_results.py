"""Plot matched independent-race scores; no publisher images are republished."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    r=json.loads((ROOT/'reports/v2-documents.json').read_text())
    names=['Numeric baseline','Document numbers','Numbers + text','Numbers + image','Full multimodal']
    scores=[r['baseline']['test']]+[r['models'][k]['test'] for k in ('numeric','numeric_text','numeric_image','multimodal')]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),layout='constrained')
    colors=['#17665c','#89999d','#799cae','#afb8b9','#afb8b9']
    for ax,key,label in zip(axes,['brier','margin_rmse'],['Brier score (lower is better)','Margin RMSE, percentage points']):
        values=[s[key] for s in scores];bars=ax.barh(names,values,color=colors,height=.65)
        ax.invert_yaxis();ax.set_xlabel(label);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,max(values)*1.28)
        for bar,v in zip(bars,values):ax.text(v+max(values)*.025,bar.get_y()+bar.get_height()/2,f'{v:.3f}' if key=='brier' else f'{v:.2f}',va='center',fontsize=10)
    fig.suptitle('Matched document test: 14 new 2024 races',fontsize=15,weight='bold')
    fig.text(.02,-.055,'Validation assigns zero document contribution to image and full branches; their scores equal the numeric baseline.\nOne retrospective cycle; these are not estimates of competition-winning odds.',fontsize=10,color='#4b5558')
    fig.savefig(ROOT/'reports/v2-ablation.png',dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
