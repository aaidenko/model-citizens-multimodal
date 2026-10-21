"""A deliberately small, time-separated fundamentals baseline, in margin points."""
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

OLD = 'https://raw.githubusercontent.com/MEDSL/constituency-returns/master/'
NEW = 'https://raw.githubusercontent.com/MEDSL/2024-elections-official/main/'


def read_csv(content):
    try:
        return pd.read_csv(BytesIO(content), encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding='latin1')


def presidential_margins(df):
    df = df.copy()
    if 'mode' in df:
        df = df[df['mode'].str.upper().eq('TOTAL')]
    party = 'party_simplified' if 'party_simplified' in df else 'party'
    votes = 'votes' if 'votes' in df else 'candidatevotes'
    df['major'] = df[party].str.lower().map({'democrat': 'D', 'republican': 'R'})
    dr = df[df['major'].notna()]
    counts = dr.pivot_table(index=['year', 'state_po'], columns='major',
                            values=votes, aggfunc='sum', fill_value=0).reset_index()
    counts['margin'] = 100 * (counts['D'] - counts['R']) / (counts['D'] + counts['R'])
    national = counts.groupby('year')[['D', 'R']].sum()
    national['national_margin'] = 100 * (national['D'] - national['R']) / (national['D'] + national['R'])
    return counts.merge(national[['national_margin']], on='year')


def midterm_backlash(year):
    # +1: Republican president, -1: Democratic president, 0: presidential election.
    if year % 4 == 0:
        return 0
    return 1 if year in {1982, 1986, 1990, 2002, 2006, 2018, 2026} else -1


def features(pres, year, state):
    previous = pres[(pres['year'] < year) & pres['state_po'].eq(state)].sort_values('year')
    if previous.empty:
        raise ValueError(f'No prior presidential result for {year} {state}')
    p = previous.iloc[-1]
    return [(p['margin'] - p['national_margin']) / 10,
            p['national_margin'] / 10, midterm_backlash(year)], int(p['year'])


def training_rows(senate, pres):
    senate = senate[senate['stage'].str.lower().eq('gen') & ~senate['special'].astype(bool)].copy()
    senate = senate[senate['state_po'].ne('LA')]  # Historical jungle-primary rules need another model.
    rows = []
    for (year, state), frame in senate.groupby(['year', 'state_po']):
        if year < 1982:
            continue
        # Sum fusion-ballot lines by candidate, using their major-party nomination.
        totals = frame.groupby('candidate')['candidatevotes'].sum()
        dem_names = set(frame.loc[frame['party'].eq('democrat'), 'candidate'])
        rep_names = set(frame.loc[frame['party'].eq('republican'), 'candidate'])
        d, r = totals.reindex(list(dem_names)).sum(), totals.reindex(list(rep_names)).sum()
        if d <= 0 or r <= 0 or (d + r) / totals.sum() < .85:
            continue  # Independents, uncontested seats, same-party top-two contests.
        x, vintage = features(pres, int(year), state)
        rows.append(dict(year=int(year), state=state, x=x,
                         margin=100 * (d - r) / (d + r), feature_year=vintage))
    return rows


def fit(rows):
    return Ridge(alpha=4).fit(np.array([r['x'] for r in rows]), [r['margin'] for r in rows])


def build(cache, states):
    senate = read_csv(cache.get('1976-2018-senate.csv', OLD + '1976-2018-senate.csv'))
    old_pres = presidential_margins(read_csv(cache.get('1976-2016-president.csv', OLD + '1976-2016-president.csv')))
    current_pres = presidential_margins(read_csv(cache.get('2024-president-state.csv', NEW + '2024-president-state.csv')))
    historical = training_rows(senate, old_pres)
    held_out = []
    for year in range(2010, 2019, 2):
        train = [r for r in historical if r['year'] < year]
        test = [r for r in historical if r['year'] == year]
        fitted = fit(train)
        train_error = np.array([r['margin'] for r in train]) - fitted.predict([r['x'] for r in train])
        sigma = max(10., float(np.sqrt(np.mean(train_error**2))))
        for r in test:
            pred = float(fitted.predict([r['x']])[0])
            held_out.append({**r, 'prediction': pred, 'q_democrat': float(norm.cdf(pred / sigma)),
                             'training_last_year': max(t['year'] for t in train)})
    residuals = np.array([r['margin'] - r['prediction'] for r in held_out])
    sigma = max(10., float(np.sqrt(np.mean(residuals**2))))
    fitted = fit(historical)
    probabilities = {}
    for st in states:
        x, vintage = features(current_pres, 2026, st)
        mean = float(fitted.predict([x])[0])
        probabilities[st] = {'mean': mean, 'sigma': sigma, 'presidential_feature_year': vintage,
                             'q_democrat': float(norm.cdf(mean / sigma))}
    y = np.array([r['margin'] > 0 for r in held_out], dtype=float)
    q = np.array([r['q_democrat'] for r in held_out])
    validation = {
        'training_rows': len(historical), 'training_last_year': 2018,
        'test_years': sorted({r['year'] for r in held_out}), 'test_races': len(held_out),
        'rmse_margin_points': sigma,
        'brier': float(np.mean((q - y)**2)),
        'log_loss': float(-np.mean(y*np.log(np.clip(q,1e-9,1))+(1-y)*np.log(np.clip(1-q,1e-9,1)))),
        'winner_accuracy': float(np.mean((q > .5) == y)),
        'held_out': held_out,
        'limitations': [
            'Validation covers the fundamentals baseline only, not the 2026 polling blend.',
            'Modern 2020/2022/2024 validation and independent-candidate modeling remain unfinished.',
            'No incumbency, candidate-quality, fundraising, or generic-ballot predictor yet.',
            'Source revisions are retrospective. Features use only prior election years, but this is not an archived real-time data-vintage backtest.',
        ],
        'coefficients': dict(zip(['state_lean_per_10pp', 'previous_national_margin_per_10pp', 'midterm_backlash'], fitted.coef_.tolist())),
        'intercept': float(fitted.intercept_),
    }
    return probabilities, validation
