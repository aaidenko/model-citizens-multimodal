"""Explicit free public-source discovery. Nothing is automatically reviewed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urljoin,urlparse

from bs4 import BeautifulSoup
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from citizens.v2_data import dump,surname
BASE=ROOT/'data/v2';RAW=BASE/'raw';RAW.mkdir(parents=True,exist_ok=True)


def fetch(url):
    key=hashlib.sha256(url.encode()).hexdigest()[:20];path=RAW/key
    if path.exists():content=path.read_bytes();stamp=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()
    else:
        r=requests.get(url,timeout=25,headers={'User-Agent':'ModelCitizensResearch/0.2 (public archive research)'})
        r.raise_for_status();content=r.content;path.write_bytes(content);stamp=datetime.now(timezone.utc).isoformat()
    return dict(url=url,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(content).hexdigest(),retrieved_at=stamp),content


def safe_fetch(url):
    try:return fetch(url)
    except Exception as e:return dict(url=url,error=type(e).__name__+': '+str(e)[:140]),b''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ppp',action='store_true',help='Explicitly enable PPP archive access where network policy permits it');args=parser.parse_args()
    snapshots=json.loads((BASE/'snapshots.json').read_text());by_race={r['race']:r for r in snapshots}
    polls=json.loads((BASE/'polls.json').read_text());candidates=[];captures=[]
    # Latest eligible publisher PDF per house/race; numeric feed already enforces
    # dates, actual nominees, one question per poll and one release per pollster.
    for p in polls:
        for url in p.get('urls',[]):
            if '.pdf' in url.lower() and not urlparse(url).hostname.endswith('fivethirtyeight.com'):
                candidates.append(dict(url=url,race=p['race'],publisher=p['pollster'],
                    published_date=p['available_date'],dem_poll=p['dem_poll'],rep_poll=p['rep_poll'],
                    dem_candidate=p['dem_candidate'],rep_candidate=p['rep_candidate'],
                    sample_size=p['sample_size'],population=p['population'],poll_id=p['poll_id'],discovery='538 original URL'))
                break
    pages=['https://www.publicpolicypolling.com/polls/?pg='+str(i) for i in range(1,35)] if args.ppp else []
    articles={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for record,content in pool.map(safe_fetch,pages):
            captures.append(record)
            soup=BeautifulSoup(content,'html.parser')
            for a in soup.select('div.col > a'):
                t=a.find('time');title=a.get_text(' ',strip=True)
                if not t or not t.get('datetime'):continue
                published=t['datetime'][:10];d=date.fromisoformat(published)
                if d.year not in (2014,2016,2018,2020,2022,2024) or d.month not in (8,9,10):continue
                eligible=[r for r in snapshots if r['year']==d.year and published<=r['forecast_date'] and
                          (date.fromisoformat(r['forecast_date'])-d).days<=90]
                if not eligible:continue
                url=urljoin(record['url'],a.get('href',''))
                articles[url]=dict(url=url,published_date=published,title=title,
                    possible_races=[r['race'] for r in eligible])
    print('Eligible PPP articles:',len(articles),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for article,(record,content) in zip(articles.values(),pool.map(safe_fetch,articles)):
            captures.append(record);soup=BeautifulSoup(content,'html.parser');text=soup.get_text(' ',strip=True).lower()
            matches=[race for race in article['possible_races'] if
                surname(by_race[race]['dem_candidate']) in text and surname(by_race[race]['rep_candidate']) in text]
            for a in soup.find_all('a',href=True):
                url=urljoin(article['url'],a['href'])
                if '.pdf' not in url.lower():continue
                for race in matches:
                    candidates.append(dict(url=url,race=race,publisher='Public Policy Polling',
                        published_date=article['published_date'],release_url=article['url'],
                        dem_candidate=by_race[race]['dem_candidate'],rep_candidate=by_race[race]['rep_candidate'],
                        discovery='PPP dated release; values require review'))
    # Reuse already downloaded, hash-reviewed original releases as candidates.
    for row in json.loads((ROOT/'data/multimodal/cases.json').read_text()):
        if row['race'] in by_race:
            candidates.append(dict(url=row['pdf_url'],race=row['race'],publisher='Marist' if 'marist' in row['pdf_url'] else 'Quinnipiac',
                published_date=row['published_date'],dem_candidate=row['dem_candidate'],rep_candidate=row['rep_candidate'],
                dem_poll=row['dem_poll'],rep_poll=row['rep_poll'],v1_review=row['id'],discovery='v0.1 reviewed original'))
    unique={}
    for r in sorted(candidates,key=lambda r:(r['published_date'],r['url'])):unique[(r['race'],r['publisher'])]=r
    candidates=list(unique.values())
    print('Candidate race/publisher documents:',len(candidates),flush=True)
    def download(c):
        record,content=safe_fetch(c['url'])
        if content and not content.startswith(b'%PDF'):record['error']='Returned bytes are not a PDF'
        return {**c,**record,'visual_review_passed':False}
    with ThreadPoolExecutor(max_workers=3) as pool:
        docs=list(pool.map(download,candidates))
    dump(BASE/'document-candidates.json',docs);dump(BASE/'discovery-captures.json',captures)
    print(json.dumps(dict(candidates=len(docs),pdfs=sum('error' not in r for r in docs),
        races=len({r['race'] for r in docs if 'error' not in r})),indent=2))


if __name__=='__main__':main()
