"""Download free original poll documents. No credentials or inference API.

Rehydrate the exact recorded bytes by default. --discover adds archive candidates;
PDFs lacking a recorded hash are research candidates, never visually approved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/multimodal'
HOSTS = {'poll.qu.edu', 'maristpoll.marist.edu', 'www.fec.gov',
         'www.uh.edu', 'raw.githubusercontent.com'}


def download(url):
    if urlparse(url).hostname not in HOSTS:
        raise ValueError('Unexpected source host')
    r = requests.get(url, timeout=40, headers={'User-Agent': 'ModelCitizensResearch/0.2'})
    r.raise_for_status()
    if urlparse(r.url).hostname not in HOSTS:
        raise ValueError('Unexpected redirect host')
    return r.content


def linked_pdfs(content, url):
    soup = BeautifulSoup(content, 'html.parser')
    chunks = [soup]
    # The publisher stores older releases as escaped HTML in public JSON-LD.
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            obj = json.loads(tag.string or tag.get_text())
            chunks.append(BeautifulSoup(obj.get('html', '').replace('\\"', '"'), 'html.parser'))
        except (ValueError, AttributeError):
            continue
    links = []
    for chunk in chunks:
        for a in chunk.find_all('a', href=True):
            link = a['href'].strip().replace('http://www.quinnipiac.edu', 'https://poll.qu.edu')
            if '.pdf' in link.lower() and not any(x in link.lower() for x in ('demo', 'trend')):
                links.append(urljoin(url, link))
    # Some legacy markup presents literal anchors inside a preformatted block.
    for suffix in re.findall(r'(?:https?://(?:poll.qu.edu|www.quinnipiac.edu))?(/images/polling/[^\s"\\<>]+\.pdf)',
                             content.decode('utf-8', errors='replace')):
        if not any(x in suffix.lower() for x in ('demo', 'trend')):
            links.append('https://poll.qu.edu' + suffix)
    return list(dict.fromkeys(links))


def discover():
    content = download('https://poll.qu.edu/Poll-Results/?date=28&items_per_page=1')
    soup = BeautifulSoup(content, 'html.parser')
    releases = {}
    for a in soup.find_all('a', href=True):
        if 'releaseid=' not in a['href'].lower() or 'read more' in a.get_text().lower():
            continue
        parent = a
        for _ in range(3):
            parent = parent.parent
        context = parent.get_text(' ', strip=True)
        date = re.match(r'(\w+ \d+, (\d{4}))', context)
        title = a.get_text(' ', strip=True)
        if not date or int(date[2]) not in (2014, 2016, 2018, 2020, 2022, 2024):
            continue
        if date[1].split()[0] not in ('August', 'September', 'October', 'November'):
            continue
        if 'senate' not in title.lower() or any(x in title.lower() for x in
                ('control senate', 'impeachment', 'supreme', 'congress', 'control of')):
            continue
        url = urljoin('https://poll.qu.edu', a['href'])
        releases[url] = dict(url=url, title=title, year=int(date[2]),
            published_date=datetime.strptime(date[1], '%B %d, %Y').date().isoformat())
    return list(releases.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--discover', action='store_true')
    args = ap.parse_args()
    BASE.mkdir(exist_ok=True, parents=True)
    (BASE / 'raw').mkdir(exist_ok=True)
    manifest_path = BASE / 'sources.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by_url = {r['url']: r for r in manifest}
    if args.discover:
        releases = discover()
        errors = []
        def get_links(row):
            try:
                # Scratch cache accelerates interrupted discovery on this machine.
                scratch = ROOT.parents[1] / 'work/multimodal-discovery' / (
                    'release-' + row['url'].split('=')[-1] + '.html')
                content = scratch.read_bytes() if scratch.exists() else download(row['url'])
                return row, linked_pdfs(content, row['url']), None
            except Exception as exc:
                return row, [], type(exc).__name__
        with ThreadPoolExecutor(max_workers=3) as pool:
            for row, links, error in pool.map(get_links, releases):
                if error:
                    errors.append(dict(url=row['url'], error=error))
                for url in links:
                    if url not in by_url:
                        by_url[url] = dict(url=url, release_url=row['url'],
                            published_date=row['published_date'], year=row['year'])
        (BASE / 'discovery.json').write_text(json.dumps(dict(
            rule='Aug-Nov Senate-titled Quinnipiac releases, 2014-2024; exclude generic control/impeachment',
            releases=releases, errors=errors), indent=2))
    def fetch(row):
        try:
            name = hashlib.sha256(row['url'].encode()).hexdigest()[:16] + Path(urlparse(row['url']).path).suffix
            path = BASE / 'raw' / name
            content = path.read_bytes() if path.exists() else download(row['url'])
            digest = hashlib.sha256(content).hexdigest()
            if row.get('sha256') and row['sha256'] != digest:
                raise ValueError('Source hash changed; stop for review')
            if row['url'].lower().endswith('.pdf') and not content.startswith(b'%PDF'):
                raise ValueError('Not a PDF')
            path.write_bytes(content)
            return {**row, 'path': str(path.relative_to(ROOT)), 'sha256': digest,
                    'retrieved_at': row.get('retrieved_at', datetime.now(timezone.utc).isoformat())}
        except Exception as exc:
            return {**row, 'error': type(exc).__name__ + ': ' + str(exc)[:120]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(fetch, by_url.values()))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(documents=len(manifest), successful=sum('error' not in r for r in manifest),
                          errors=[r for r in manifest if 'error' in r]), indent=2))


if __name__ == '__main__':
    main()
