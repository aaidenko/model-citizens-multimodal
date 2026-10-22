"""Locate additional publisher-rendered legacy reports for browser review."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from citizens.v2_data import dump,surname
from collect_v2_documents import safe_fetch,BASE


def main():
    rows=json.loads((BASE/'snapshots.json').read_text());old={r['race'] for r in json.loads((ROOT/'data/multimodal/cases.json').read_text())}
    lookup={r['race']:r for r in rows if r['year']<=2018 and r['race'] not in old}
    url='https://poll.qu.edu/Poll-Results/?date=28&items_per_page=1';capture,content=safe_fetch(url);captures=[capture]
    soup=BeautifulSoup(content,'html.parser');releases={}
    for a in soup.find_all('a',href=True):
        if 'releaseid=' not in a['href'].lower():continue
        p=a
        for _ in range(3):p=p.parent
        m=re.match(r'(\w+ \d+, (\d{4}))',p.get_text(' ',strip=True))
        if not m or int(m[2]) not in range(1998,2019,2) or m[1].split()[0] not in ('August','September','October') or 'senate' not in a.get_text().lower():continue
        published=datetime.strptime(m[1],'%B %d, %Y').date().isoformat()
        eligible=[r for r in lookup.values() if r['year']==int(m[2]) and published<=r['forecast_date']]
        if eligible:releases[urljoin(url,a['href'])]=dict(published_date=published,eligible=eligible)
    chosen={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for (link,release),(record,content) in zip(releases.items(),pool.map(safe_fetch,releases)):
            captures.append(record);soup=BeautifulSoup(content,'html.parser')
            for tag in soup.find_all('script',type='application/ld+json'):
                try:obj=json.loads(tag.string or tag.get_text())
                except ValueError:continue
                for question in obj.get('pollQuestions',[]):
                    q=question.get('Question','');qlo=q.lower()
                    if 'senat' not in qlo or 'election' not in qlo:continue
                    for r in release['eligible']:
                        if surname(r['dem_candidate']) not in qlo or surname(r['rep_candidate']) not in qlo:continue
                        # The same surname can refer to candidates in different
                        # states. Both nominees and the actual table get checked.
                        case=dict(id=r['race']+'-quinnipiac-html',race=r['race'],year=r['year'],state=r['state'],
                            published_date=release['published_date'],release_url=link,question=q,
                            dem_candidate=r['dem_candidate'],rep_candidate=r['rep_candidate'],
                            html_path=record['path'],html_sha256=record['sha256'])
                        if r['race'] not in chosen or case['published_date']>chosen[r['race']]['published_date']:chosen[r['race']]=case
    dump(BASE/'html-review-plan.json',list(chosen.values()));dump(BASE/'html-sources.json',captures)
    print(json.dumps({'eligible_releases':len(releases),'additional_races':len(chosen),'races':sorted(chosen)},indent=2))


if __name__=='__main__':main()
