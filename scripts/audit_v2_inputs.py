"""Report outcome coverage and image geometry without granting source approval."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.v2_data import dump,sha

if __name__=='__main__':
    rows=json.loads((ROOT/'data/v2/snapshots.json').read_text());cases=json.loads((ROOT/'data/v2/cases.json').read_text());labels=json.loads((ROOT/'data/v2/labels.json').read_text())
    def counts(ids):
        values=[labels[x]['margin']>0 for x in ids]
        return dict(races=len(values),democratic_wins=sum(values),republican_wins=len(values)-sum(values))
    dump(ROOT/'reports/v2-cohort-audit.json',dict(
        numeric={k:counts([r['race'] for r in rows if r['split']==k]) for k in ('train','development','calibration','test','consumed')},
        documents={k:counts([r['race'] for r in cases if condition(r)]) for k,condition in [
            ('initial_train',lambda r:r['year']<=2018),('development',lambda r:r['year']==2020),
            ('final_fit',lambda r:r['year']<=2020),('test',lambda r:r['year']==2024)]},
        limitations=['All five document-development outcomes are Republican wins. This remains a serious coverage limit, not evidence that winner imbalance caused any particular error.',
            'The larger numeric development set includes both parties. Do not describe the document development set as representative.',
            'The new test is now consumed; do not change selection to improve its score.']))
    path=ROOT/'data/v2/embeddings-manifest.json';manifest=json.loads(path.read_text());records=manifest['records']
    auditpath=ROOT/'reports/v2-image-audit.json';prior=json.loads(auditpath.read_text()) if auditpath.exists() else {}
    verified=prior.get('manifest_sha256')==sha(path)
    dump(auditpath,dict(status='REPAIRED IN VERSIONED v0.2 INPUTS',cases=len(records),
        all_first_views_retain_complete_source=all(r['first_view_retained_area_fraction']==1 for r in records),
        minimum_views=min(r['views'] for r in records),maximum_views=max(r['views'] for r in records),additional_center_crop=False,
        manual_actual_tensor_preview_checks=prior.get('manual_actual_tensor_preview_checks',[]) if verified else [],
        checks=prior.get('checks','Actual encoder previews require visual inspection.') if verified else 'Changed manifest requires renewed actual-input visual inspection.',
        manifest_sha256=sha(path)))
    print('Outcome coverage and image-geometry audits refreshed; no new visual approval granted.')
