"""Restore omitted publisher originals; verify exact recorded hashes.

Mutable historical URLs may have changed. Fail rather than silently substitute
new bytes into an old backtest. Training from included embeddings stays offline.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import requests

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {'raw.githubusercontent.com', 'www.tpsielections.com', 'external-api.kalshi.com',
         'www.uh.edu', 'texaspolitics.utexas.edu', 'poll.qu.edu', 'maristpoll.marist.edu', 'www.fec.gov'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true', help='Include baseline poll/market capture originals')
    args = ap.parse_args()
    manifests = [ROOT/'data/multimodal/sources.json', ROOT/'data/multimodal/labels-source.json',
                 ROOT/'data/multimodal/labels-source-2022.json']
    if args.all:
        manifests.append(ROOT/'data/sources.json')
    records = []
    for path in manifests:
        obj = json.loads(path.read_text())
        records.extend(obj if isinstance(obj, list) else [obj])
    # Include current original document used by the encoded demonstration.
    originals = json.loads((ROOT/'data/sources.json').read_text())
    records += [r for r in originals if r['id'] == 'hobby-2026.pdf']
    failed = []
    for row in {r['path']: r for r in records if 'path' in r}.values():
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT/'data') or urlparse(row['url']).hostname not in HOSTS:
            raise ValueError('Unexpected manifest destination or host')
        try:
            if path.exists():
                content = path.read_bytes()
            else:
                response = requests.get(row['url'], timeout=40)
                response.raise_for_status()
                if urlparse(response.url).hostname not in HOSTS:
                    raise ValueError('Unexpected redirect host')
                content = response.content
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Source changed from recorded hash')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except Exception as exc:
            failed.append(dict(path=row['path'], error=str(exc)[:140]))
    print(json.dumps(dict(failed=failed, checked=len(records)), indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
