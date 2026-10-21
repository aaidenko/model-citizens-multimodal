import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from citizens.documents import extract_hobby
from citizens.forecast import aggregate_polls, validate_market
from citizens.scoring import choose, expected_points, reward, underdog_threshold
from citizens.sources import Cache

ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    def test_reward_cap_and_examples(self):
        self.assertAlmostEqual(reward(.25),2)
        self.assertAlmostEqual(reward(.9),.152003093445)
        self.assertEqual(reward(0),reward(.01))
        self.assertEqual(reward(.001),reward(.01))
        self.assertEqual(reward(1),0)

    def test_threshold_switches_decision(self):
        threshold=underdog_threshold(.1)
        self.assertAlmostEqual(threshold,.0437553557,places=8)
        self.assertEqual(choose([threshold-.001,1-threshold+.001],[.1,.9]),1)
        self.assertEqual(choose([threshold+.001,1-threshold-.001],[.1,.9]),0)

    def test_market_agreement_can_favor_underdog(self):
        self.assertEqual(choose([.1,.9],[.1,.9]),0)
        self.assertGreater(expected_points(.1,.1),2*expected_points(.9,.9))

    def test_benchmark_prices_are_not_normalized(self):
        self.assertAlmostEqual(expected_points(.5,.41),-.5*math.log2(.41))
        self.assertEqual(choose([.2,.5,.3],[.05,.7,.25]),0)

    def test_invalid_probabilities_rejected(self):
        for p in [-.01,1.01,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): reward(p)
            with self.assertRaises(ValueError): expected_points(p,.5)
        with self.assertRaises(ValueError): choose([.2,.5],[.5,.5])


class DataSafetyTests(unittest.TestCase):
    def test_settlement_rules_control_mapping(self):
        raw=json.loads((ROOT/'data/raw/LA-markets.json').read_text())
        m=next(m for m in raw['markets'] if m['event_ticker'].endswith('-26'))
        self.assertEqual(validate_market(m,'LA'),'settlement_rules_state_mismatch')
        self.assertIsNone(validate_market(m,'KY'))

    def test_future_old_and_duplicate_polls_excluded(self):
        p=dict(pollster='Example',date='2026-10-10',a=48,b=46,sampleSize=800,sampleType='LV')
        result,_=aggregate_polls({'polls':[p,p,dict(p,date='2026-10-22'),dict(p,date='2025-10-01')]},date(2026,10,21))
        self.assertEqual(result['pollsters'],1)
        self.assertEqual(len(result['observations']),1)
        self.assertEqual(result['mean'],2)
        self.assertGreaterEqual(result['sigma'],4)

    def test_repeated_pollster_does_not_multiply_independent_evidence(self):
        p=dict(pollster='Example',date='2026-10-10',a=48,b=46,sampleSize=800,sampleType='LV')
        result,_=aggregate_polls({'polls':[p,dict(p,date='2026-10-12',a=49)]},date(2026,10,21))
        self.assertEqual(result['pollsters'],1)
        self.assertEqual(result['mean'],3)

    def test_cache_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data/raw').mkdir(parents=True)
            (root/'data/raw/a.json').write_text('{}')
            (root/'data/sources.json').write_text(json.dumps([dict(id='a.json',url='https://www.uh.edu/a.json',path='data/raw/a.json',sha256='wrong')]))
            with self.assertRaises(ValueError): Cache(root).get('a.json','https://www.uh.edu/a.json')

    def test_paid_or_unapproved_host_not_contacted(self):
        with patch('requests.get',side_effect=AssertionError('Network must not be called')):
            with self.assertRaises(ValueError):
                Cache(ROOT).get('x.json','https://api.example-paid-provider.com/x')

    def test_offline_cache_missing_does_not_contact_network(self):
        with patch('requests.get',side_effect=AssertionError('Offline mode must never contact network')):
            with self.assertRaises(ValueError):
                Cache(ROOT,offline=True).get('not-cached.json','https://www.uh.edu/not-cached.json')

    def test_future_capture_not_used_for_past_forecast(self):
        with patch('requests.get',side_effect=AssertionError('Past-date replay must stay offline')):
            with self.assertRaises(ValueError):
                Cache(ROOT,offline=True,as_of=date(2026,10,20)).get(
                    'hobby-2026.pdf','https://www.uh.edu/hobby/texas2026/statewide26.pdf')

    def test_document_hash_is_required_for_visual_verification(self):
        with patch('requests.get',side_effect=AssertionError('Must use cached public source')):
            evidence=extract_hobby(Cache(ROOT,offline=True),'not-the-verified-hash')
        self.assertTrue(evidence['chart_table_agree'])
        self.assertFalse(evidence['visual_review_passed'])
        self.assertEqual(evidence['poll_observation']['date'],'2026-10-14')
        self.assertEqual(evidence['poll_observation']['sampleSize'],2000)

    def test_public_clone_replays_reviewed_facts_without_publisher_pdf(self):
        original=json.loads((ROOT/'data/evidence/hobby-reviewed.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data/evidence').mkdir(parents=True)
            (root/'data/evidence/hobby-reviewed.json').write_text(json.dumps(original))
            (root/'data/sources.json').write_text(json.dumps([dict(id='hobby-2026.pdf',
                url='https://www.uh.edu/hobby/texas2026/statewide26.pdf',
                path='data/raw/omitted.pdf',sha256=original['sha256'])]))
            with patch('requests.get',side_effect=AssertionError('Offline clone contacted network')):
                self.assertEqual(extract_hobby(Cache(root,offline=True),original['sha256']),original)
                changed=dict(original,sha256='changed-source')
                (root/'data/evidence/hobby-reviewed.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    extract_hobby(Cache(root,offline=True),original['sha256'])

    def test_pdf_poll_is_not_counted_twice(self):
        config=json.loads((ROOT/'config.json').read_text())
        evidence=extract_hobby(Cache(ROOT,offline=True),config['visually_verified_hobby_pdf_sha256'])
        old=dict(evidence['poll_observation'],pollster='University of Houston',date='2026-10-15')
        result,_=aggregate_polls({'polls':[old]},date(2026,10,21),evidence)
        self.assertEqual(result['pollsters'],1)
        self.assertEqual(result['observations'][0]['date'],'2026-10-14')


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((ROOT/'reports/research.json').read_text())

    def test_template_coverage_and_probability_mass(self):
        self.assertEqual(len(self.report['races']),35)
        self.assertEqual(len({r['race_id'] for r in self.report['races']}),35)
        for r in self.report['races']:
            if r['outcomes']:
                self.assertAlmostEqual(sum(o['q'] for o in r['outcomes']),1)
                self.assertEqual(sum(o['selected'] for o in r['outcomes']),1)

    def test_no_submission_ready_claim_before_benchmark(self):
        self.assertFalse(self.report['submission_ready'])
        self.assertTrue(any('benchmark' in b for b in self.report['blockers']))
        blocked={r['race_id'] for r in self.report['races'] if not r['outcomes']}
        self.assertEqual(blocked,{'LA-SEN','KY-SEN'})
        with (ROOT/'drafts/predictions-RESEARCH-ONLY.csv').open() as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            if r['race_id'] in blocked:self.assertEqual(r['candidate_name'],'')

    def test_history_never_uses_same_year_presidential_result(self):
        for r in self.report['validation']['held_out']:
            self.assertLess(r['feature_year'],r['year'])
            self.assertLess(r['training_last_year'],r['year'])

    def test_unpolled_tails_use_explicit_market_proxy(self):
        for r in self.report['races']:
            if r['outcomes'] and not r['poll_summary']:
                self.assertEqual(r['model_method'],'market_proxy')
                for o in r['outcomes']:
                    self.assertAlmostEqual(o['q'],o['q_market_proxy'])

    def test_simulator_splits_first_place_credit(self):
        for rho in [0,.25,.5]:
            rows=[r for r in self.report['scenario_comparison']['results'] if r['latent_correlation']==rho]
            self.assertAlmostEqual(sum(r['first_place_share'] for r in rows),1,places=8)

    def test_full_replay_requires_no_network_and_is_reproducible(self):
        import run
        path=ROOT/'reports/research.json'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with patch('sys.argv',['run.py','--as-of','2026-10-21']),patch('requests.get',side_effect=AssertionError('Offline replay requested')):
            run.main()
        self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main()
