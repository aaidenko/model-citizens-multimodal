"""Leakage-conscious, offline historical poll snapshots. Labels stay separate."""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

CONSUMED = {'2024-MI', '2024-PA', '2024-WI'}
FEATURES = ['poll_margin', 'remaining', 'dispersion', 'staleness', 'log_pollsters',
            'log_sample', 'sample_missing', 'partisan_share', 'online_share',
            'state_lean', 'previous_national', 'midterm', 'returning_winner']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def clean(value):
    return re.sub(r'[^a-z ]', '', str(value).lower()).strip()


def surname(value):
    words = [w for w in clean(value).split() if w not in {'jr', 'sr', 'ii', 'iii', 'iv'}]
    return words[-1] if words else ''


def major_returns(frame, year_col='year'):
    party = 'party_simplified' if 'party_simplified' in frame else 'party'
    votes = 'votes' if 'votes' in frame else 'candidatevotes'
    out = {}
    for (year, state), f in frame.groupby([year_col, 'state_po']):
        if 'mode' in f:
            f = f[f['mode'].str.upper().eq('TOTAL')]
        if 'stage' in f:
            f = f[f.stage.str.lower().eq('gen')]
        if 'special' in f:
            f = f[~f.special.astype(str).str.lower().eq('true')]
        totals = f.groupby('candidate')[votes].sum()
        names = {}
        for short, key in [('D', 'democrat'), ('R', 'republican')]:
            candidates = set(f.loc[f[party].str.lower().eq(key), 'candidate'])
            if candidates:
                names[short] = totals.reindex(list(candidates)).idxmax()
        if len(names) != 2:
            continue
        d, r = (float(totals[names[k]]) for k in ('D', 'R'))
        if min(d, r) <= 0 or (d+r)/max(float(totals.sum()), 1) < .85:
            continue
        out[(int(year), state)] = dict(margin=100*(d-r)/(d+r),
            dem_candidate=str(names['D']), rep_candidate=str(names['R']),
            winner=str(names['D'] if d > r else names['R']))
    return out


def history_inputs(root, sources):
    """Only prior results supply presidential/returning-winner features."""
    from citizens.fundamentals import presidential_margins
    old = pd.read_csv(root/'data/raw/1976-2016-president.csv', encoding='latin1')
    pres = presidential_margins(old).to_dict('records')
    raw = pd.read_csv(sources/'raw_polls.csv')
    p = raw[raw.type_simple.eq('Pres-G') & raw.cycle.eq(2020) &
            raw.cand1_party.eq('DEM') & raw.cand2_party.eq('REP')]
    national = p[p.location.eq('US')][['cand1_actual', 'cand2_actual']].median()
    nat = 100*(national.iloc[0]-national.iloc[1])/national.sum()
    for st, f in p[~p.location.eq('US')].groupby('location'):
        d, r = f[['cand1_actual', 'cand2_actual']].median()
        pres.append(dict(year=2020, state_po=st, margin=100*(d-r)/(d+r), national_margin=nat))
    old_labels = major_returns(pd.read_csv(root/'data/raw/1976-2018-senate.csv', encoding='latin1'))
    return raw, pres, old_labels


def historical_polls(raw, old_labels):
    polls, labels, exclusions = [], {}, []
    frame = raw[raw.type_simple.eq('Sen-G') & raw.cand1_party.eq('DEM') & raw.cand2_party.eq('REP')]
    for (year, st, rid), f in frame.groupby(['cycle', 'location', 'race_id']):
        year = int(year); race = f'{year}-{st}'
        election = date.fromisoformat(str(f.electiondate.iloc[0]))
        reason = None
        if year % 2 or election.month != 11 or st == 'LA':
            reason = 'odd-year/special, runoff date, or Louisiana jungle election'
        if (year, st) in {(2008, 'GA'), (2020, 'GA'), (2022, 'GA'), (2020, 'AZ'), (2020, 'AK')}:
            reason = 'runoff-dependent general, special election, or independent nominee'
        if year <= 2018:
            eligible = old_labels.get((year, st))
            if not eligible or surname(f.cand1_name.iloc[0]) != surname(eligible['dem_candidate']) or surname(f.cand2_name.iloc[0]) != surname(eligible['rep_candidate']):
                reason = 'not the eligible regular D/R race in MEDSL'
        elif int(rid) in {9480, 9482}:
            reason = '2022 California duplicate special or Oklahoma special'
        if reason:
            exclusions.append(dict(race=race, source_race_id=int(rid), reason=reason));continue
        label = old_labels.get((year, st))
        if label is None:
            d, r = f[['cand1_actual', 'cand2_actual']].median()
            label = dict(margin=100*(d-r)/(d+r), dem_candidate=str(f.cand1_name.iloc[0]),
                rep_candidate=str(f.cand2_name.iloc[0]), winner=str(f.cand1_name.iloc[0] if d > r else f.cand2_name.iloc[0]))
        labels[race] = {**label, 'source': 'MEDSL regular returns' if year <= 2018 else '538 archived actual vote shares', 'year':year}
        for _, row in f.iterrows():
            field = date.fromisoformat(row.polldate); available = field+timedelta(days=7)
            if not election-timedelta(days=104) <= field or available > election-timedelta(days=14):
                continue
            polls.append(dict(race=race, year=year, state=st, election_date=election.isoformat(),
                poll_id=str(row.poll_id), question_id=str(row.question_id), pollster=str(row.pollster),
                field_date=field.isoformat(),field_date_kind='median_field_date', available_date=available.isoformat(),
                availability='median field date + 7 days (proxy)', dem_poll=float(row.cand1_pct),
                rep_poll=float(row.cand2_pct), dem_candidate=str(row.cand1_name), rep_candidate=str(row.cand2_name),
                sample_size=None if pd.isna(row.samplesize) else float(row.samplesize), population=None,
                mode='' if pd.isna(row.methodology) else str(row.methodology),
                partisan=False if pd.isna(row.partisan) else True, source='538-ratings'))
    return polls, labels, exclusions


def polls_2024(path, state_map):
    f = pd.read_csv(path)
    f = f[f.cycle.eq(2024) & f.stage.eq('general') & ~f.hypothetical.fillna(True).astype(bool) &
          ~f.ranked_choice_reallocated.fillna(False).astype(bool) & f.seat_name.eq('Class I')]
    f = f[~f.state.isin(['Maine','Vermont','Nebraska'])]
    f['end'] = pd.to_datetime(f.end_date, format='%m/%d/%y')
    f['created'] = pd.to_datetime(f.created_at, format='%m/%d/%y %H:%M')
    cutoff = pd.Timestamp('2024-10-22')
    f = f[(f.end <= cutoff) & (f.created < cutoff) & (f.end >= cutoff-pd.Timedelta(days=90))]
    # An outcome-free roster proxy: most recent eligible non-hypothetical nominee
    # per party. Persist selected identities for review rather than use results.
    roster = f[f.party.isin(['DEM','REP'])].sort_values(['end','created','candidate_id']).groupby(['state','party']).tail(1)
    nominees = {(r.state,r.party):int(r.candidate_id) for r in roster.itertuples()}
    out = []
    for qid, g in f.groupby('question_id'):
        st = str(g.state.iloc[0]); candidates = {}
        for party in ('DEM','REP'):
            rows = g[g.party.eq(party) & g.candidate_id.eq(nominees.get((st,party), -1))]
            if len(rows)==1:candidates[party]=rows.iloc[0]
        if len(candidates)!=2:continue
        d,r=candidates['DEM'],candidates['REP'];ab=state_map[st.upper()]
        if not 0<float(d.pct)<100 or not 0<float(r.pct)<100 or float(d.pct+r.pct)>100:continue
        out.append(dict(race='2024-'+ab, year=2024, state=ab, election_date='2024-11-05',
            poll_id=str(d.poll_id), question_id=str(qid), pollster=str(d.pollster),
            field_date=d.end.date().isoformat(),field_date_kind='reported_field_end', available_date=(d.created.date()+timedelta(days=1)).isoformat(),
            availability='archived ingestion date + one-day timezone guard', dem_poll=float(d.pct),
            rep_poll=float(r.pct), dem_candidate=str(d.candidate_name), rep_candidate=str(r.candidate_name),
            sample_size=None if pd.isna(d.sample_size) else float(d.sample_size), population=str(d.population),
            mode='' if pd.isna(d.methodology) else str(d.methodology),
            partisan=False if pd.isna(d.partisan) else True, source='538-2024-archive',
            urls=list(dict.fromkeys(str(d[k]) for k in ('url_crosstab','url_topline','url','url_article') if pd.notna(d[k])))))
    return out


def aggregate(polls, pres, labels):
    from citizens.fundamentals import midterm_backlash
    groups = {}
    for p in polls:
        if min(p['dem_poll'],p['rep_poll'])<=0 or p['dem_poll']+p['rep_poll']>100:continue
        groups.setdefault(p['race'], []).append(p)
    snapshots=[];kept=[]
    for race, rows in sorted(groups.items()):
        year=rows[0]['year'];st=rows[0]['state'];cutoff=date.fromisoformat(rows[0]['election_date'])-timedelta(days=14)
        if race not in labels:continue
        by_poll={}
        for p in sorted(rows,key=lambda p:(p['field_date'],p['available_date'],{'lv':3,'rv':2,'a':1}.get(p['population'],0),p['question_id'])):
            if date.fromisoformat(p['available_date'])>cutoff:continue
            if (cutoff-date.fromisoformat(p['field_date'])).days>90:continue
            by_poll[p['poll_id']]=p
        by_house={}
        for p in sorted(by_poll.values(),key=lambda p:(p['field_date'],p['available_date'],p['poll_id'])):by_house[p['pollster']]=p
        selected=list(by_house.values())
        if not selected:continue
        prior=[p for p in pres if p['state_po']==st and p['year']<year]
        if not prior:continue
        prior=max(prior,key=lambda p:p['year'])
        age=np.array([(cutoff-date.fromisoformat(p['field_date'])).days for p in selected])
        size=np.array([p['sample_size'] or 600 for p in selected])
        weights=np.sqrt(np.clip(size,100,1500))*np.exp2(-age/21)
        weights/=weights.sum()
        margins=np.array([100*(p['dem_poll']-p['rep_poll'])/(p['dem_poll']+p['rep_poll']) for p in selected])
        margin=float(weights@margins)
        previous=labels.get(f'{year-6}-{st}')
        returning=0
        if previous:
            winner=surname(previous['winner'])
            if winner==surname(selected[0]['dem_candidate']):returning=1
            elif winner==surname(selected[0]['rep_candidate']):returning=-1
        values=[margin,float(weights@np.array([100-p['dem_poll']-p['rep_poll'] for p in selected])),
            float(np.sqrt(weights@((margins-margin)**2))),float(weights@age),float(np.log1p(len(selected))),
            float(weights@np.log(np.maximum(size,1))),float(weights@np.array([p['sample_size'] is None for p in selected])),
            float(weights@np.array([p['partisan'] for p in selected])),float(weights@np.array(['Online' in p['mode'] for p in selected])),
            float(prior['margin']-prior['national_margin']),float(prior['national_margin']),float(midterm_backlash(year)),float(returning)]
        snapshots.append(dict(race=race,year=year,state=st,forecast_date=cutoff.isoformat(),
            presidential_feature_year=int(prior['year']),returning_winner_feature_year=year-6 if previous else None,
            feature_names=FEATURES,x=values,poll_ids=[p['poll_id'] for p in selected],
            dem_candidate=selected[0]['dem_candidate'],rep_candidate=selected[0]['rep_candidate'],
            split='consumed' if race in CONSUMED else 'test' if year==2024 else 'calibration' if year==2022 else 'development' if year in (2018,2020) else 'train'))
        kept.extend(selected)
    return snapshots,kept


def build(root, source_dir=None):
    root=Path(root);base=root/'data/v2';dest=base/'sources';dest.mkdir(parents=True,exist_ok=True)
    source_dir=Path(source_dir) if source_dir else dest
    for name in ('raw_polls.csv','senate-2024.csv'):
        if (source_dir/name).resolve()!=(dest/name).resolve():shutil.copyfile(source_dir/name,dest/name)
    sources=[]
    for filename,meta in [('raw_polls.csv','raw_polls.source.json'),('senate-2024.csv','senate-2024.csv.source.json')]:
        if (source_dir/meta).exists():
            record=json.loads((source_dir/meta).read_text());shutil.copyfile(source_dir/meta,dest/meta) if (source_dir/meta).resolve()!=(dest/meta).resolve() else None
        else:record=json.loads((dest/meta).read_text())
        sources.append(dict(record,path=str((dest/filename).relative_to(root)),sha256=sha(dest/filename)))
    raw,pres,old_labels=history_inputs(root,dest)
    polls,labels,exclusions=historical_polls(raw,old_labels)
    labels.update({f'{y}-{s}':{**r,'year':y,'source':'MEDSL regular returns'} for (y,s),r in old_labels.items() if y<1998})
    returns=pd.read_csv(root/'data/raw/2024-senate-state.csv');current_labels=major_returns(returns)
    state_map={str(r.state).upper():str(r.state_po) for r in pd.read_csv(root/'data/raw/2024-president-state.csv').itertuples()}
    fresh=polls_2024(dest/'senate-2024.csv',state_map)
    polls.extend(fresh)
    for (y,st),label in current_labels.items():labels[f'{y}-{st}']={**label,'year':y,'source':'MEDSL 2024 state returns; some rows marked unofficial'}
    snapshots,kept=aggregate(polls,pres,labels)
    ids={r['race'] for r in snapshots}
    dump(base/'snapshots.json',snapshots);dump(base/'polls.json',kept)
    dump(base/'labels.json',{k:v for k,v in labels.items() if k in ids})
    dump(base/'sources.json',sources);dump(base/'exclusions.json',exclusions)
    summary=dict(races=len(snapshots),polls=len(kept),features=FEATURES,
        splits={k:sum(r['split']==k for r in snapshots) for k in ('train','development','calibration','test','consumed')},
        years={str(y):sum(r['year']==y for r in snapshots) for y in sorted({r['year'] for r in snapshots})},
        source_sha256={Path(s['path']).name:s['sha256'] for s in sources},
        files={name:sha(base/name) for name in ('snapshots.json','polls.json','labels.json')})
    dump(base/'manifest.json',summary)
    return summary
