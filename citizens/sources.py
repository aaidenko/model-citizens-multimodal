import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

ALLOWED_HOSTS = {
    'external-api.kalshi.com', 'raw.githubusercontent.com',
    'www.tpsielections.com', 'www.uh.edu', 'texaspolitics.utexas.edu',
}


class Cache:
    """GET-only cache, with original bytes, capture times, and SHA-256 hashes."""

    def __init__(self, root, refresh=False, offline=False, as_of=None):
        self.root = Path(root)
        self.refresh = refresh
        self.offline = offline
        self.as_of = as_of
        self.index = self.root / 'data/sources.json'
        self.records = json.loads(self.index.read_text()) if self.index.exists() else []
        self._lock = Lock()

    def get(self, source_id, url):
        if urlparse(url).hostname not in ALLOWED_HOSTS:
            raise ValueError(f'Unapproved source host: {url}')
        if Path(source_id).name != source_id:
            raise ValueError('Source IDs must be plain filenames')
        previous = self._eligible(source_id)
        if previous and not self.refresh:
            rec = previous[-1]
            if rec['url'] != url:
                raise ValueError(f'Source URL changed for {source_id}')
            content = (self.root / rec['path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != rec['sha256']:
                raise ValueError(f'Cached source hash mismatch: {source_id}')
            return content
        if self.offline:
            raise ValueError(f'No eligible cached capture for {source_id}; offline replay cannot download data')
        today = datetime.now(ZoneInfo('America/Los_Angeles')).date()
        if self.as_of is not None and self.as_of < today:
            raise ValueError('A new download cannot establish what was available on a past date')
        response = requests.get(url, timeout=35, headers={'User-Agent': 'ModelCitizensResearch/0.1'})
        response.raise_for_status()
        content = response.content
        if source_id.endswith('.json'):
            json.loads(content)
        if source_id.endswith('.pdf') and not content.startswith(b'%PDF'):
            raise ValueError('Source did not return a PDF')
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(source_id).suffix
        path = Path('data/raw') / f'{Path(source_id).stem}-{digest[:12]}{suffix}'
        (self.root / path).write_bytes(content)
        rec = dict(id=source_id, url=url, path=str(path), sha256=digest,
                   retrieved_at=datetime.now(timezone.utc).isoformat())
        with self._lock:
            self.records.append(rec)
            self.index.write_text(json.dumps(self.records, indent=2))
        return content

    def json(self, source_id, url):
        return json.loads(self.get(source_id, url))

    def record(self, source_id):
        return self._eligible(source_id)[-1]

    def _eligible(self, source_id):
        records = [r for r in self.records if r['id'] == source_id]
        if self.as_of is not None:
            records = [r for r in records if datetime.fromisoformat(r['retrieved_at']).astimezone(
                ZoneInfo('America/Los_Angeles')).date() <= self.as_of]
        return records
