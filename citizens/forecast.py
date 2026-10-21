from datetime import date
import math
import re
import unicodedata

import numpy as np
from scipy.stats import norm

from .scoring import choose, expected_points

STATES = dict(AK='Alaska', AL='Alabama', AR='Arkansas', CO='Colorado', DE='Delaware',
 FL='Florida', GA='Georgia', IA='Iowa', ID='Idaho', IL='Illinois', KS='Kansas',
 KY='Kentucky', LA='Louisiana', MA='Massachusetts', ME='Maine', MI='Michigan',
 MN='Minnesota', MS='Mississippi', MT='Montana', NC='North Carolina', NE='Nebraska',
 NH='New Hampshire', NJ='New Jersey', NM='New Mexico', OH='Ohio', OK='Oklahoma',
 OR='Oregon', RI='Rhode Island', SC='South Carolina', SD='South Dakota',
 TN='Tennessee', TX='Texas', VA='Virginia', WV='West Virginia', WY='Wyoming')


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if c.isalnum())


def party(market):
    suffix = market['ticker'].rsplit('-', 1)[-1]
    if suffix in {'D', 'R'}:
        return suffix
    if 'independent' in market.get('subtitle', '').lower() or suffix == 'IND':
        return 'I'
    return 'OTHER'


def quote(market):
    bid, ask = float(market['yes_bid_dollars']), float(market['yes_ask_dollars'])
    if not (0 <= bid <= ask <= 1) or ask == 0:
        raise ValueError('Missing, inverted, or invalid quote')
    return (bid + ask)/2, ask-bid


def validate_market(market, state):
    if not market['event_ticker'].endswith('-26'):
        return 'wrong_election_year'
    if market.get('status') not in {'active', 'open'}:
        return 'market_not_active'
    rules = market.get('rules_primary', '')
    if not re.search(r'\b'+re.escape(STATES[state])+r'\b', rules, re.I):
        return 'settlement_rules_state_mismatch'
    if not ('2026' in rules or '2027' in rules):
        return 'settlement_year_unverified'
    try:
        quote(market)
    except (ValueError, KeyError):
        return 'invalid_quote'
    return None


def match_outcome(label, markets):
    needle = canonical(label)
    hits = [i for i,m in enumerate(markets)
            if needle and needle in canonical(m.get('yes_sub_title', ''))]
    return hits[0] if len(hits) == 1 else None


def aggregate_polls(data, as_of, evidence=None):
    rows, issues, seen = [], [], set()
    for p in data.get('polls', []):
        try:
            day = date.fromisoformat(p['date'])
            age = (as_of-day).days
            a,b = float(p['a']), float(p['b'])
            if age < 0 or age > 90:
                continue
            if not (0 <= a <= 100 and 0 <= b <= 100 and a+b <= 101):
                issues.append('invalid_poll_shares'); continue
            key = (canonical(p['pollster']), p['date'], a, b)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(p, a=a, b=b))
        except (ValueError, KeyError, TypeError):
            issues.append('unparseable_poll')
    if evidence and evidence['visual_review_passed']:
        official = evidence['poll_observation']
        # The source report is the same poll already in the aggregate: replace it.
        rows = [p for p in rows if not (
            ('houston' in p['pollster'].lower() or 'hobby' in p['pollster'].lower())
            and abs((date.fromisoformat(p['date'])-date.fromisoformat(official['date'])).days) <= 3)]
        if 0 <= (as_of-date.fromisoformat(official['date'])).days <= 90:
            rows.append(official)
    if not rows:
        return None, issues + ['no_poll_in_last_90_days']
    # One recent observation per pollster avoids treating repeated releases as independent.
    latest = {}
    for p in sorted(rows, key=lambda p:p['date']):
        latest[canonical(p['pollster'])] = p
    rows = list(latest.values())
    weights, margins = [], []
    for p in rows:
        age = (as_of-date.fromisoformat(p['date'])).days
        sample = float(p.get('sampleSize') or 600)
        w = 2**(-age/21) * min(math.sqrt(max(sample,1)/600), 2)
        w *= {'LV':1, 'RV':.8, 'A':.6}.get(p.get('sampleType'), .6)
        weights.append(w); margins.append(p['a']-p['b'])
    weights = np.asarray(weights); weights /= weights.sum()
    mean = float(np.dot(weights, margins))
    newest_age = (as_of-max(date.fromisoformat(p['date']) for p in rows)).days
    dispersion = float(np.dot(weights,(np.asarray(margins)-mean)**2))
    effective_n = 1 / float(np.sum(weights**2))
    # Shared polling error does not vanish with more polls. These are exploratory settings.
    sigma = math.sqrt(4.0**2 + dispersion/max(effective_n,1) + (newest_age*.12)**2)
    return dict(mean=mean, sigma=sigma, pollsters=len(rows), polls_eligible=len(rows),
                latest_date=max(p['date'] for p in rows), observations=rows,
                uncertainty_settings='4pp shared-error floor; 21-day half-life; not historically calibrated'), issues


def build_race(state, bundle, poll_bundle, fundamental, as_of, evidence=None):
    issues, markets = [], []
    for m in bundle['markets']:
        err = validate_market(m, state)
        if err:
            issues.append(f'{err}: {m["ticker"]}')
        else:
            markets.append(m)
    if not markets:
        issues.append('no_verified_market')
    # An event is an outcome set; partial acceptance could wrongly exclude the winner.
    if len(markets) != len(bundle['markets']) or len(markets) < 2:
        markets = []
    poll, poll_issues = (None, [])
    if poll_bundle:
        if evidence:
            avg=poll_bundle['data'].get('avg',{})
            labels=[canonical(avg.get(k,{}).get('label','')) for k in ['a','b']]
            if labels != ['talarico','paxton']:
                evidence=None
                issues.append('document_candidate_order_unverified')
        poll, poll_issues = aggregate_polls(poll_bundle['data'], as_of, evidence)
    issues.extend(poll_issues)
    result = dict(race_id=state+'-SEN', state=state, state_name=STATES[state],
                  issues=issues, outcomes=[], model_method='unavailable',
                  poll_summary=poll, fundamental=fundamental, market_url=bundle['url'],
                  poll_url=poll_bundle['url'] if poll_bundle else None)
    if not markets:
        return result
    prices = [quote(m)[0] for m in markets]
    q_market = np.asarray(prices) / sum(prices)
    q = q_market.copy()
    parties = [party(m) for m in markets]
    method = 'market_proxy'
    # Do not turn an old, broad fundamentals error distribution into precise
    # present-day long-shot probabilities where there are no recent polls.
    # Preserve that model as a diagnostic; use the explicitly labelled market
    # proxy until modern-cycle tail calibration is available.
    if poll_bundle and poll:
        avg = poll_bundle['data']['avg']
        a = match_outcome(avg['a']['label'], markets)
        b = match_outcome(avg['b']['label'], markets)
        # Alaska's party-labelled contracts need an explicit, sourced name override.
        if state == 'AK' and {avg['a']['label'], avg['b']['label']} == {'Peltola','Sullivan'}:
            if set(parties)=={'D','R'}:
                a = parties.index('D' if avg['a']['label']=='Peltola' else 'R')
                b = parties.index('R' if avg['b']['label']=='Sullivan' else 'D')
                issues.append('candidate_names_need_full_roster_verification')
        if a is not None and b is not None and a != b:
            if parties[a]=='D' and parties[b]=='R':
                prior_mean, prior_sigma = fundamental['mean'], fundamental['sigma']
            elif parties[a]=='R' and parties[b]=='D':
                prior_mean, prior_sigma = -fundamental['mean'], fundamental['sigma']
            else:
                prior_mean, prior_sigma = 0., 20.
                issues.append('independent_race_neutral_prior_not_calibrated')
            variance = 1/(1/prior_sigma**2+1/poll['sigma']**2)
            mean = variance*(prior_mean/prior_sigma**2+poll['mean']/poll['sigma']**2)
            q_a = float(norm.cdf(mean/math.sqrt(variance)))
            residual_mass = sum(q_market[i] for i in range(len(markets)) if i not in {a,b})
            q = q_market.copy(); q[a] = (1-residual_mass)*q_a; q[b] = (1-residual_mass)*(1-q_a)
            method = 'polls_plus_fundamentals' if {parties[a],parties[b]}=={'D','R'} else 'polls_with_neutral_independent_prior'
            if residual_mass:
                issues.append('minor_outcome_probabilities_use_market_proxy')
        else:
            issues.append('poll_candidate_mapping_unverified')
    if method == 'market_proxy':
        issues.append('no_independent_forecast_for_this_race')
        issues.append('fundamentals_tail_model_not_promoted_pending_modern_validation')
    selected = choose(q.tolist(), prices)
    for i,m in enumerate(markets):
        name = m['yes_sub_title']
        if name.lower() in {'democratic party','republican party'}:
            issues.append('generic_party_label_instead_of_candidate_name')
        result['outcomes'].append(dict(
            ticker=m['ticker'], party=parties[i], candidate=name,
            benchmark_proxy=prices[i], spread=quote(m)[1],
            q=float(q[i]), q_market_proxy=float(q_market[i]),
            expected_points=expected_points(float(q[i]),prices[i]),
            selected=i==selected, rules=m['rules_primary'],
        ))
    result['selected_index'] = selected
    result['model_method'] = method
    result['issues'] = sorted(set(issues))
    return result
