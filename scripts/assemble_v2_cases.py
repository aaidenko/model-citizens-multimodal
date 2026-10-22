"""Assemble hash-pinned approved facts; never infer visual approval from a parse."""
from datetime import date,timedelta
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.v2_data import dump,sha,surname
BASE=ROOT/'data/v2'


def main():
    cases=json.loads((BASE/'pdf-approved-cases.json').read_text())
    plan={r['id']:r for r in json.loads((BASE/'html-review-plan.json').read_text())}
    approved=json.loads((BASE/'html-visual-reviews.json').read_text())
    for review in approved:
        r=plan[review['id']];dompath=BASE/'raw'/(r['id']+'-dom.json');image=BASE/'pages'/(r['id']+'.png')
        if sha(ROOT/r['html_path'])!=review['html_sha256'] or sha(image)!=review['image_sha256'] or sha(dompath)!=review['dom_sha256']:raise ValueError('Changed source or screenshot requires visual review')
        dom=json.loads(dompath.read_text());published=date.fromisoformat(r['published_date'])
        if datetime_date(dom['date'])!=published:raise ValueError('Publisher date changed')
        sample=review.get('sample_size');population=review.get('population')
        # The encoder text is restricted to the historical question and first
        # table. No current sidebar, recommendations, navigation or footer.
        text=r['question']+'\n'+'\n'.join(' | '.join(row['cells']) for row in dom['rows'][:20])
        textpath=BASE/'text'/(r['id']+'.txt');textpath.parent.mkdir(parents=True,exist_ok=True);textpath.write_text(text)
        snap=next(s for s in json.loads((BASE/'snapshots.json').read_text()) if s['race']==r['race'])
        election=date.fromisoformat(snap['forecast_date'])+timedelta(days=14)
        cases.append(dict(id=r['id'],race=r['race'],year=r['year'],state=r['state'],published_date=r['published_date'],
            election_date=election.isoformat(),publisher='Quinnipiac',source_kind='publisher_html_screenshot',
            release_url=r['release_url'],html_path=r['html_path'],html_sha256=review['html_sha256'],
            image_path=str(image.relative_to(ROOT)),image_sha256=review['image_sha256'],
            crop_pixels=review['crop_pixels'],text_path=str(textpath.relative_to(ROOT)),text_sha256=sha(textpath),
            dem_candidate=r['dem_candidate'],rep_candidate=r['rep_candidate'],dem_poll=review['dem_poll'],rep_poll=review['rep_poll'],
            sample_size=sample,population=population,sample_size_verified=False,
            sample_size_source='Native publisher methodology text; visual approval covers topline table only',
            review_method=review['review_method'],review_notes=review.get('notes',''),visual_review_passed=True,layout_holdout=False))
    cases.sort(key=lambda r:(r['year'],r['state'],r['id']))
    dump(BASE/'cases.json',cases)
    print(json.dumps(dict(cases=len(cases),independent_races=len({r['race'] for r in cases}),
        historical_training_races=len({r['race'] for r in cases if r['year']<=2020}),
        new_test_races=len({r['race'] for r in cases if r['year']==2024})),indent=2))


def datetime_date(value):
    from datetime import datetime
    return datetime.strptime(value,'%B %d, %Y').date()


if __name__=='__main__':main()
