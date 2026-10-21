from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from citizens.multimodal import read_cases, numeric, split_indices, design, aggregate, digest

ROOT = Path(__file__).resolve().parents[1]


class MultimodalIntegrity(unittest.TestCase):
    def setUp(self):
        self.cases = read_cases(ROOT)

    def test_elections_are_separated_and_current_outcome_is_unknown(self):
        train, valid, test, current = split_indices(self.cases)
        self.assertEqual({self.cases[i]['year'] for i in train}, {2014, 2016, 2018, 2020})
        self.assertEqual({self.cases[i]['year'] for i in valid}, {2022})
        self.assertEqual({self.cases[i]['year'] for i in test}, {2024})
        self.assertTrue(all(self.cases[i]['actual_dem_margin'] is None for i in current))
        self.assertEqual(len({r['race'] for r in self.cases}), len(self.cases))

    def test_labels_cannot_affect_features(self):
        row = deepcopy(self.cases[0])
        original = numeric(row)
        row['actual_dem_margin'] = 99
        self.assertEqual(numeric(row), original)

    def test_numeric_values_are_major_party_conditional(self):
        row = dict(dem_poll=48, rep_poll=46, election_date='2026-11-03',
                   published_date='2026-10-21', likely_voters=True)
        self.assertAlmostEqual(numeric(row)[0], 100*(48-46)/(48+46)/10)

    def test_candidate_identity_collision_is_resolved(self):
        murphy = [r for r in self.cases if 'Murphy' in r['dem_candidate']]
        self.assertEqual({(r['year'], r['state']) for r in murphy}, {(2016, 'FL'), (2018, 'CT')})

    def test_pca_and_scaling_ignore_holdout_values(self):
        tr, va, te, cur = split_indices(self.cases)
        rng = np.random.default_rng(17)
        emb = {k: rng.normal(size=(len(self.cases), 12)) for k in ('text', 'image')}
        first, _, _ = design(self.cases, emb, tr)
        for name in emb:
            emb[name][va+te+cur] *= 1000
        second, _, _ = design(self.cases, emb, tr)
        np.testing.assert_array_equal(first[tr], second[tr])

    def test_repeated_poll_does_not_inflate_independent_race_count(self):
        row = deepcopy(self.cases[0])
        later = deepcopy(row)
        later['published_date'] = '2014-10-20'
        result = aggregate([row, later], [0, 1], [1, 2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['predicted_dem_margin'], 20)

    def test_unreviewed_or_current_labeled_case_is_rejected(self):
        for field, value, index in [('visual_review_passed', False, 0),
                                    ('actual_dem_margin', 10, -1)]:
            cases = deepcopy(self.cases)
            cases[index][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'data/multimodal'
                path.mkdir(parents=True)
                (path/'cases.json').write_text(json.dumps(cases))
                with self.assertRaises(ValueError):
                    read_cases(directory)

    def test_visual_reviews_pin_the_same_source_and_crop(self):
        reviews = {r['id']: r for r in json.loads((ROOT/'data/multimodal/visual-reviews.json').read_text())}
        for row in self.cases:
            review = reviews[row['id']]
            self.assertEqual((review['pdf_sha256'], review['crop'], review['page']),
                             (row['pdf_sha256'], row['crop'], row['page']))
            self.assertEqual((review['dem_poll'], review['rep_poll']), (row['dem_poll'], row['rep_poll']))

    def test_learned_artifacts_are_pinned_and_not_promoted(self):
        run = json.loads((ROOT/'models/multimodal/run.json').read_text())
        for name, expected in run['artifacts'].items():
            self.assertEqual(digest(ROOT/'models/multimodal'/name), expected)
        self.assertFalse(run['promotion']['promoted'])
        self.assertGreater(run['models']['multimodal']['parameter_l2_change_last_seed_at_max_epoch'], 0)
        self.assertGreater(run['models']['multimodal']['modality_sensitivity']['image_masked_mean_abs_margin_change'], 0)
        self.assertGreater(run['models']['multimodal']['modality_sensitivity']['text_masked_mean_abs_margin_change'], 0)


if __name__ == '__main__':
    unittest.main()
