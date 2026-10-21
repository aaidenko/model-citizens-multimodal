from concurrent.futures import ThreadPoolExecutor
import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

POLL_HOST = 'https://www.tpsielections.com'
KALSHI = 'https://external-api.kalshi.com/trade-api/v2'


def collect_polls(cache):
    # Public clones omit publisher HTML. Replay our discovery metadata while
    # still verifying every included numeric feed against its original hash.
    index_record = cache.record('senate-polls-index.html') if cache._eligible('senate-polls-index.html') else None
    if cache.offline and index_record and not (cache.root/index_record['path']).exists():
        bundles = json.loads((cache.root/'data/poll-bundles.json').read_text())
        return [{**b, 'data': cache.json(b['source_id'], b['url'])}
                for b in bundles if 'source_id' in b]
    index = cache.get('senate-polls-index.html', POLL_HOST + '/polls/senate')
    soup = BeautifulSoup(index, 'html.parser')
    pages = sorted({a['href'] for a in soup.find_all('a', href=True)
                    if re.fullmatch(r'/polls/[a-z]{2}/senate', a['href'])})

    def one(page):
        st = page.split('/')[2].upper()
        page_id = f'{st}-poll-page.html'
        html = cache.get(page_id, POLL_HOST + page)
        doc = BeautifulSoup(html, 'html.parser')
        link = next((a.get('href') for a in doc.find_all('a')
                     if a.get_text(strip=True) == 'Public JSON'), None)
        if not link:
            return {'state': st, 'error': 'No documented public JSON link'}
        url = urljoin(POLL_HOST, link)
        data_id = f'{st}-polls.json'
        result = cache.json(data_id, url)
        return {'state': st, 'source_id': data_id, 'url': url, 'data': result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, pages))


def collect_markets(cache, states):
    def one(st):
        series = 'SENATE' + st + ('S' if st in {'FL', 'OH'} else '')
        source_id = f'{st}-markets.json'
        url = KALSHI + '/markets?series_ticker=' + series + '&limit=1000'
        data = cache.json(source_id, url)
        if data.get('cursor'):
            raise ValueError(f'Unconsumed market pagination: {st}')
        return {'state': st, 'source_id': source_id, 'url': url,
                'markets': [m for m in data['markets']
                            if m['event_ticker'].endswith('-26')]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, states))
