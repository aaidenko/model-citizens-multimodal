"""Diagnose v0.1's default square center crop without changing its features.

The retained area estimate assumes CLIP's configured shortest-edge resize and
square center crop. It is a geometry diagnostic, not an accuracy prediction.
It also works on public clones using the recorded original PDF crop bounds.
"""
from pathlib import Path
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def audit():
    cases = json.loads((ROOT/'data/multimodal/cases.json').read_text())
    rows = []
    for case in cases:
        path = ROOT/'data/multimodal/pages'/f'{case["id"]}.png'
        if path.exists():
            with Image.open(path) as image:
                width, height = image.size
            geometry_source = 'rendered source image pixels'
        elif case.get('crop'):
            x0, y0, x1, y1 = case['crop']
            width, height = x1-x0, y1-y0
            geometry_source = 'recorded original PDF crop points'
        else:
            continue
        rows.append(dict(id=case['id'], year=case['year'],
            geometry_source=geometry_source, width=width, height=height,
            estimated_area_retained=min(width, height)/max(width, height)))
    training = [r for r in rows if r['year'] <= 2020]
    return dict(experiment='v0.1; unchanged',
        assumption='shortest-edge resize followed by square center crop, as configured in the pinned CLIP processor',
        interpretation='Approximate original image area retained, before integer resize rounding. Not a measured accuracy effect.',
        training_images=len(training),
        training_images_losing_over_half_the_area=sum(r['estimated_area_retained'] < .5 for r in training),
        minimum_training_area_retained=min(r['estimated_area_retained'] for r in training),
        examples=rows)


if __name__ == '__main__':
    output = audit()
    destination = ROOT/'reports/image-input-audit.json'
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps({k:v for k,v in output.items() if k != 'examples'}, indent=2))
