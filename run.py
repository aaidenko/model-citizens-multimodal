#!/usr/bin/env python3
"""Replay the public-data research snapshot. --refresh makes free public GET requests."""
import argparse
import csv
from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from citizens.collect import collect_markets, collect_polls
from citizens.documents import extract_hobby
from citizens.forecast import build_race
from citizens.fundamentals import build as build_fundamentals
from citizens.report import render
from citizens.simulate import compare
from citizens.sources import Cache

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--as-of', default='2026-10-21')
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of)
    cache = Cache(ROOT, refresh=args.refresh, offline=not args.refresh, as_of=as_of)
    with (ROOT/'data/senate-template.csv').open() as f:
        template = list(csv.DictReader(f))
    states = [r['race_id'].split('-')[0] for r in template]
    if len(states) != 35 or len(set(states)) != 35:
        raise ValueError('Expected the 35 distinct race IDs from the organizer template')
    print('Collecting/replaying public polling and market snapshots...', flush=True)
    markets = collect_markets(cache, states)
    polls = collect_polls(cache)
    # A historical year filter cannot turn today's snapshot into a historical archive.
    used_ids = {b['source_id'] for b in markets} | {p['source_id'] for p in polls if 'source_id' in p}
    for source_id in used_ids:
        observed = datetime.fromisoformat(cache.record(source_id)['retrieved_at']).astimezone(ZoneInfo('America/Los_Angeles')).date()
        if as_of < observed:
            raise ValueError(f'{source_id} was first observed after the requested forecast date')
    config = json.loads((ROOT/'config.json').read_text())
    evidence = extract_hobby(cache, config['visually_verified_hobby_pdf_sha256'])
    fundamentals, validation = build_fundamentals(cache, states)
    poll_by_state = {b['state']:b for b in polls if 'data' in b}
    races = [build_race(b['state'], b, poll_by_state.get(b['state']),
                        fundamentals[b['state']], as_of,
                        evidence if b['state']=='TX' else None) for b in markets]
    comparison = compare(races)
    comparison['excluded_races'] = [r['race_id'] for r in races if not r['outcomes']]
    blockers = [
        'Official November 1 benchmark has not been supplied.',
        'Full polling-blend calibration and 2020/2022/2024 validation are unfinished.',
        'All candidate names, party codes, and ballot status need final independent roster verification.',
        'Confirm treatment of the CSV probability field and the submission deadline timezone.',
    ]
    missing = [r['race_id'] for r in races if not r['outcomes']]
    if missing:
        blockers.append('Unverified/missing market mappings: '+', '.join(missing))
    if not evidence['visual_review_passed']:
        blockers.append('Changed PDF bytes require fresh page-image review.')
    multimodal_path = ROOT/'reports/multimodal.json'
    multimodal = json.loads(multimodal_path.read_text()) if multimodal_path.exists() else None
    audit_path = ROOT/'reports/image-input-audit.json'
    image_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    report = dict(as_of=as_of.isoformat(), status='RESEARCH DRAFT — NOT READY TO SUBMIT',
                  new_service_spend_usd=0, submission_ready=False, blockers=blockers,
                  counts=dict(races=len(races), verified_market_races=len(races)-len(missing),
                              poll_feeds=len(polls),
                              raw_poll_rows=sum(len(p['data'].get('polls',[])) for p in polls if 'data' in p)),
                  races=races, validation=validation, document_evidence=evidence,
                  scenario_comparison=comparison, multimodal_experiment=multimodal,
                  multimodal_input_audit=image_audit)
    (ROOT/'reports/research.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (ROOT/'reports/validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False))
    # Blank rows are intentional: an unresolved outcome set must never silently get a pick.
    with (ROOT/'drafts/predictions-RESEARCH-ONLY.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(template[0]))
        writer.writeheader()
        for race in races:
            o = next((o for o in race['outcomes'] if o['selected']),None)
            writer.writerow(dict(race_id=race['race_id'],chamber='Senate',
                predicted_winner_party=o['party'] if o else '',
                candidate_name=o['candidate'] if o else '',
                win_probability=f'{o["q"]:.6f}' if o else ''))
    (ROOT/'reports/submission-readiness.json').write_text(json.dumps({
        'ready':False,'blockers':blockers,'unresolved_races':missing,
        'draft_csv':'drafts/predictions-RESEARCH-ONLY.csv',
        'multimodal': {'trained': multimodal is not None, 'promoted': False,
            'reason': 'Small held-out experiment underperformed matched numeric-only model; see reports/multimodal.json.',
            'known_input_issue': 'Default square image cropping removes important columns in some wide tables. Repair is planned for a versioned experiment; see reports/image-input-audit.json.',
            'next_experiment_plan': 'docs/IMPROVEMENT_PLAN.md'}},indent=2))
    render(ROOT, report)
    print(json.dumps({'status':report['status'],'counts':report['counts'],
                      'fundamentals_validation_brier':validation['brier'],
                      'report':str(ROOT/'reports/index.html')},indent=2))


if __name__=='__main__':
    main()
