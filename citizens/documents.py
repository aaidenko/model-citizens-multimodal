"""Document evidence: preserve the page, extract tables, reconcile repeated totals."""
import hashlib
from io import BytesIO
import json
import re

import pdfplumber

HOBBY = 'https://www.uh.edu/hobby/texas2026/statewide26.pdf'
TRACKER = 'https://texaspolitics.utexas.edu/blog/texas-2026-u-s-senate-poll-tracker'
# This exact PDF's page images were visually inspected in the initial build.
# A changed source hash must go through visual review again.


def extract_hobby(cache, verified_hash):
    # Reproduce the already reviewed extraction in public clones, where the
    # publisher's original bytes are deliberately omitted. New/changed captures
    # cannot use this fallback. No new visual review is implied by replay.
    if cache.offline:
        record = cache.record('hobby-2026.pdf')
        if not (cache.root/record['path']).exists():
            frozen = json.loads((cache.root/'data/evidence/hobby-reviewed.json').read_text())
            if record['url'] != HOBBY or frozen['sha256'] != record['sha256']:
                raise ValueError('Missing original PDF does not match the frozen reviewed source')
            frozen['visual_review_passed'] = frozen['sha256'] == verified_hash
            return frozen
    content = cache.get('hobby-2026.pdf', HOBBY)
    digest = hashlib.sha256(content).hexdigest()
    with pdfplumber.open(BytesIO(content)) as doc:
        method = doc.pages[1].extract_text()
        chart = doc.pages[5].extract_text()
        tables = doc.pages[6].extract_tables()
        table = next(t for t in tables if t and 'Talarico' in str(t[0]))
        overall = next(r for r in table if r[0] == 'Overall')
        values = [float(v) for v in overall[2:6]]
        chart_values = []
        for label in ['James Talarico (D)', 'Ken Paxton (R)', 'Ted Brown (L)', 'Unsure']:
            found = re.search(re.escape(label) + r'\s+(\d+)\b', chart)
            if not found:
                raise ValueError(f'Cannot reconcile chart label: {label}')
            chart_values.append(float(found.group(1)))
        if chart_values != values or abs(sum(values)-100) > .01:
            raise ValueError('PDF chart and table totals do not agree')
        dates = re.search(r'October (\d+) and October (\d+), 2026', method)
        n = re.search(r'population of ([\d,]+) Texas likely', method)
        if not dates or not n:
            raise ValueError('Cannot locate poll methodology')
        subgroup_rows = []
        for row in table:
            if len(row) >= 6 and row[1] and all(v and str(v).isdigit() for v in row[2:6]):
                subgroup_rows.append({'group': row[1], 'values': [int(v) for v in row[2:6]]})
    return {
        'source_url': HOBBY, 'sha256': digest,
        'source_pages_1_based': {'methodology': 2, 'chart': 6, 'table': 7},
        'poll_observation': {'pollster': 'University of Houston / Hobby School',
            'date': f'2026-10-{int(dates.group(2)):02d}',
            'field_start': f'2026-10-{int(dates.group(1)):02d}',
            'sampleSize': int(n.group(1).replace(',', '')), 'sampleType': 'LV',
            'a': values[0], 'b': values[1], 'third_party': values[2], 'undecided': values[3]},
        'chart_table_agree': True, 'visual_review_passed': digest == verified_hash,
        'chart_image': '../data/evidence/hobby-06.png',
        'table_image': '../data/evidence/hobby-07.png',
        'subgroups': subgroup_rows,
        'predictive_use': 'One topline poll only, after hash-specific visual review. Subgroup/visual features remain excluded pending historical validation.',
        'issue': 'UT tracker lists October 10-15; original PDF methodology says October 10-14. Use the original field dates.',
        'tracker_url': TRACKER,
    }
