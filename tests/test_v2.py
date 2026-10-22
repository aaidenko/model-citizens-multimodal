from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import unittest
from importlib.util import find_spec

import numpy as np
from PIL import Image

from citizens.image_inputs import source_views
from citizens.v2_data import sha,FEATURES,CONSUMED,aggregate,history_inputs
from citizens.v2_model import scaler,fit_calibration
from citizens.v2_documents import doc_design,read_cases
from citizens.local_ocr import row_value

ROOT=Path(__file__).resolve().parents[1]


class V2Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/'data/v2'
        cls.rows=json.loads((cls.base/'snapshots.json').read_text())
        cls.polls=json.loads((cls.base/'polls.json').read_text())
        cls.labels=json.loads((cls.base/'labels.json').read_text())
        cls.cases=read_cases(ROOT)

    def test_source_and_derived_bytes_are_pinned(self):
        manifest=json.loads((self.base/'manifest.json').read_text())
        for name,h in manifest['files'].items():self.assertEqual(sha(self.base/name),h)
        for r in json.loads((self.base/'sources.json').read_text()):self.assertEqual(sha(ROOT/r['path']),r['sha256'])

    def test_independent_outcomes_and_temporal_partitions(self):
        self.assertGreater(len(self.rows),300)
        self.assertEqual(len(self.rows),len({r['race'] for r in self.rows}))
        for r in self.rows:
            self.assertLess(r['presidential_feature_year'],r['year'])
            if r['returning_winner_feature_year'] is not None:self.assertEqual(r['returning_winner_feature_year'],r['year']-6)
            self.assertEqual(len(r['x']),len(FEATURES))
            self.assertTrue(np.isfinite(r['x']).all())
            if r['split']=='test':
                self.assertEqual(r['year'],2024);self.assertNotIn(r['race'],CONSUMED)
            if r['split']=='calibration':self.assertEqual(r['year'],2022)

    def test_latest_per_pollster_and_availability_cutoff(self):
        by_race={}
        for p in self.polls:by_race.setdefault(p['race'],[]).append(p)
        for r in self.rows:
            polls=by_race[r['race']]
            self.assertEqual(len(polls),len({p['pollster'] for p in polls}))
            self.assertEqual(set(r['poll_ids']),{p['poll_id'] for p in polls})
            for p in polls:
                self.assertLessEqual(p['available_date'],r['forecast_date'])
                self.assertLessEqual((date.fromisoformat(r['forecast_date'])-date.fromisoformat(p['field_date'])).days,90)
                self.assertIn(p['field_date_kind'],{'median_field_date','reported_field_end'})
                self.assertNotIn('actual_dem_margin',p)

    def test_current_or_future_labels_cannot_change_features(self):
        _,pres,_=history_inputs(ROOT,self.base/'sources')
        original,_=aggregate(self.polls,pres,self.labels)
        altered=deepcopy(self.labels)
        for k,r in altered.items():
            if r['year']>=2022:r.update(margin=-99,winner='Fabricated future winner')
        second,_=aggregate(self.polls,pres,altered)
        self.assertEqual(original,second)
        injected=deepcopy(self.polls)
        for p in injected:p.update(margin_actual=99,numeric_grade=99,transparency_score=99)
        third,_=aggregate(injected,pres,self.labels)
        self.assertEqual(original,third)

    def test_constant_features_do_not_activate_untrained_weights(self):
        x=np.array([[1,0],[3,0],[5,9999]],dtype=float)
        transformed,_,scale=scaler(x,[0,1])
        self.assertEqual(scale[1],0)
        np.testing.assert_array_equal(transformed[:,1],[0,0,0])

    def test_image_preprocessing_retains_left_and_right_columns(self):
        im=Image.new('RGB',(800,160),'white')
        for x,color in [(0,'red'),(768,'blue')]:im.paste(color,(x,0,x+32,160))
        views,boxes=source_views(im)
        self.assertEqual(boxes[0],(0,0,800,160))
        self.assertEqual(views[0].size,(224,224))
        self.assertGreater(views[0].getpixel((3,112))[0],200)
        self.assertGreater(views[0].getpixel((220,112))[2],200)
        self.assertGreater(len(views),2)

    def test_document_preprocessing_fits_only_earlier_cycles(self):
        cases=self.cases;train=[i for i,r in enumerate(cases) if r['year']<=2018]
        rng=np.random.default_rng(17);emb={k:rng.normal(size=(len(cases),8)) for k in ('text','image')}
        baseline=np.zeros(len(cases));first,saved=doc_design(cases,emb,train,baseline)
        for name in emb:
            for i,r in enumerate(cases):
                if r['year']>2018:emb[name][i]*=1000
        second,again=doc_design(cases,emb,train,baseline)
        for a,b in zip(first,second):np.testing.assert_array_equal(a[train],b[train])
        for name in saved:np.testing.assert_array_equal(saved[name],again[name])

    def test_calibration_rejects_test_year_and_ignores_unused_labels(self):
        pred=np.array([-10,-5,0,5,10,20],dtype=float);y=np.array([-9,-7,1,6,12,22],dtype=float);years=[2022]*5+[2024]
        first=fit_calibration(pred,y,list(range(5)),years)
        y[-1]=-999;second=fit_calibration(pred,y,list(range(5)),years)
        self.assertEqual(first,second)
        with self.assertRaises(ValueError):fit_calibration(pred,y,[0,5],years)

    def test_document_hashes_reviews_and_outcome_counts(self):
        manifest=json.loads((self.base/'embeddings-manifest.json').read_text())
        self.assertEqual(sha(self.base/'cases.json'),manifest['cases_sha256'])
        self.assertEqual(sha(self.base/'embeddings.npz'),manifest['embeddings_sha256'])
        self.assertEqual(len(self.cases),len({r['race'] for r in self.cases}))
        self.assertGreaterEqual(len(self.cases),50)
        for r in manifest['records']:
            self.assertEqual(r['first_view_retained_area_fraction'],1)
            self.assertFalse(r['additional_center_crop'])
        for r in self.cases:self.assertNotIn('actual_dem_margin',r)

    def test_v1_artifacts_are_preserved(self):
        checkpoint=json.loads((ROOT/'reports/pause-checkpoint.json').read_text())
        for name,h in checkpoint['preserved_v1_artifact_hashes'].items():self.assertEqual(sha(ROOT/name),h)

    def test_training_happened_and_promotion_is_honest(self):
        numeric=json.loads((ROOT/'reports/v2-numeric.json').read_text());docs=json.loads((ROOT/'reports/v2-documents.json').read_text())
        self.assertEqual(numeric['final_fit_last_year'],2020)
        for trace in numeric['neural_training_trace']:self.assertGreater(trace['parameter_l2_change'],0)
        for model in docs['models'].values():
            for trace in model['training_trace']:self.assertGreater(trace['parameter_l2_change'],0)
        self.assertFalse(docs['promotion']['promoted'])
        for model in docs['models'].values():
            if model['selected']['weight']==0:self.assertEqual(model['test'],docs['baseline']['test'])
        for name,h in docs['artifacts'].items():self.assertEqual(sha(ROOT/'models/v2'/name),h)

    def test_ocr_never_guesses_missing_or_wrong_row_values(self):
        words=[dict(text='Example',left=0,right=50,top=10,bottom=20),
            dict(text='48%',left=90,right=120,top=10,bottom=20),
            dict(text='99%',left=60,right=80,top=80,bottom=90)]
        self.assertEqual(row_value(words,'Example'),48)
        self.assertIsNone(row_value(words,'Missing'))
        self.assertIsNone(row_value(words[:1]+words[2:],'Example'))

    @unittest.skipUnless(find_spec('torch'),'Optional training environment required')
    def test_saved_weights_reproduce_predictions_without_fitting(self):
        from citizens.v2_inference import predict
        run=json.loads((ROOT/'reports/v2-documents.json').read_text())
        for variant in run['models']:
            expected=next(r for r in run['models'][variant]['test_predictions'] if r['race']=='2024-OH')
            actual=predict(ROOT,'2024-OH-the-washington-post',variant)
            self.assertAlmostEqual(actual['predicted_dem_margin'],expected['predicted_dem_margin'],places=5)
            self.assertAlmostEqual(actual['q_democrat'],expected['q_democrat'],places=6)


if __name__=='__main__':unittest.main()
