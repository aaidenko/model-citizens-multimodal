"""Scenario comparisons, not estimates of the real contest's win probability."""
import numpy as np
from scipy.stats import norm

from .scoring import reward


def compare(races, simulations=20000, seed=20261021):
    races = [r for r in races if r['outcomes']]
    policies = {name: [] for name in ['Forecast favorites', 'Market favorites',
                                      'Market expected points', 'Forecast expected points',
                                      'Conservative expected points']}
    for r in races:
        o = r['outcomes']
        q = np.array([x['q'] for x in o])
        market = np.array([x['q_market_proxy'] for x in o])
        weights = np.array([reward(x['benchmark_proxy']) for x in o])
        for name, values in [
            ('Forecast favorites', q), ('Market favorites', market),
            ('Market expected points', market*weights),
            ('Forecast expected points', q*weights),
            ('Conservative expected points', np.minimum(q,market)*weights),
        ]:
            policies[name].append(int(np.argmax(values)))
    results = []
    for rho in [0., .25, .5]:
        rng = np.random.default_rng(seed)
        common = rng.normal(size=simulations)
        scores = np.zeros((simulations,len(policies)))
        for k,r in enumerate(races):
            # This ordering makes a favorable shared shock benefit D/independent outcomes.
            order = sorted(range(len(r['outcomes'])), key=lambda i:
                           {'D':0,'I':1,'R':2,'OTHER':3}[r['outcomes'][i]['party']])
            cumulative = np.cumsum([r['outcomes'][i]['q'] for i in order]); cumulative[-1]=1.
            latent = np.sqrt(rho)*common + np.sqrt(1-rho)*rng.normal(size=simulations)
            outcome = np.array(order)[np.searchsorted(cumulative,norm.cdf(latent),side='left')]
            for j,picks in enumerate(policies.values()):
                pick = picks[k]
                scores[:,j] += (outcome==pick)*reward(r['outcomes'][pick]['benchmark_proxy'])
        leaders = np.isclose(scores,scores.max(axis=1,keepdims=True),rtol=0,atol=1e-10)
        share = leaders/leaders.sum(axis=1,keepdims=True)
        for j,name in enumerate(policies):
            results.append(dict(policy=name, latent_correlation=rho,
                                mean_points=float(scores[:,j].mean()),
                                first_place_share=float(share[:,j].mean()),
                                score_10th_percentile=float(np.quantile(scores[:,j],.1)),
                                score_90th_percentile=float(np.quantile(scores[:,j],.9))))
    return dict(simulations=simulations, seed=seed, races_included=len(races),
                excluded_races=[], results=results,
                assumptions=[
                    'Only races with verified market mappings are included.',
                    'Current bid/ask midpoint prices stand in for the unavailable November 1 benchmark.',
                    'The provisional forecast is assumed true in these simulations.',
                    'Shared latent correlations are sensitivity assumptions, not fitted parameters.',
                    'Five hypothetical entries, one per policy; ties split first-place credit equally.',
                    'These numbers are not actual competition winning probabilities.',
                ])
