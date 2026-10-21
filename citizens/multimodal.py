"""Local image + text + tabular election experiment. See docs/MODEL_CARD.md.

The January 2021 CLIP encoders stay frozen. We train our forecasting network,
including its image/text inputs, using real historical Senate election labels.
Nothing in this module calls a hosted inference service.
"""
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ENCODER_ID = 'openai/clip-vit-base-patch32'
ENCODER_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
SEEDS = (17, 29, 43)
EPOCHS = (50, 150, 300)
VARIANTS = ('numeric', 'numeric_text', 'numeric_image', 'multimodal')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(root):
    cases = json.loads((Path(root) / 'data/multimodal/cases.json').read_text())
    ids = set()
    for r in cases:
        if r['id'] in ids:
            raise ValueError('Duplicate case ID')
        ids.add(r['id'])
        if r['year'] != int(r['published_date'][:4]):
            raise ValueError('Publication/election year mismatch')
        if not (0 < r['dem_poll'] < 100 and 0 < r['rep_poll'] < 100 and
                r['dem_poll'] + r['rep_poll'] <= 100):
            raise ValueError('Invalid topline')
        if r['year'] < 2026 and r.get('actual_dem_margin') is None:
            raise ValueError('Missing historical outcome')
        if r['year'] == 2026 and r.get('actual_dem_margin') is not None:
            raise ValueError('Current outcome leakage')
        if not r.get('visual_review_passed'):
            raise ValueError('Case has not received visual review')
        if date.fromisoformat(r['published_date']) >= date.fromisoformat(r['election_date']):
            raise ValueError('Post-election document')
        days = (date.fromisoformat(r['election_date']) - date.fromisoformat(r['published_date'])).days
        if r['year'] < 2026 and not 14 <= days <= 100:
            raise ValueError('Historical document outside the frozen forecast window')
        if r['race'] != f'{r["year"]}-{r["state"]}':
            raise ValueError('Race/state/year identity mismatch')
    return cases


def numeric(row):
    margin = 100 * (row['dem_poll'] - row['rep_poll']) / (row['dem_poll'] + row['rep_poll'])
    days = (date.fromisoformat(row['election_date']) - date.fromisoformat(row['published_date'])).days
    return [margin / 10, (100-row['dem_poll']-row['rep_poll']) / 10,
            days / 90, float(row.get('likely_voters', True))]


def prepare(root, encoder_path):
    """Embed actual source PDF crops and text; local encoder files are required."""
    import pdfplumber
    from PIL import Image
    import torch
    from transformers import CLIPModel, CLIPProcessor

    root = Path(root)
    cases = read_cases(root)
    encoder_path = Path(encoder_path)
    if not encoder_path.is_dir():
        raise ValueError('Provide downloaded local CLIP directory; no implicit downloads')
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained(str(encoder_path), local_files_only=True).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(str(encoder_path), local_files_only=True)
    images, texts, records = [], [], []
    for i, row in enumerate(cases):
        source = root / row['pdf_path']
        if digest(source) != row['pdf_sha256']:
            raise ValueError(f'Changed PDF requires review: {row["id"]}')
        image_path = root / 'data/multimodal/pages' / (row['id'] + '.png')
        image_path.parent.mkdir(exist_ok=True, parents=True)
        with pdfplumber.open(source) as pdf:
            page = pdf.pages[row['page']-1]
            crop = page.crop(tuple(row['crop'])) if row.get('crop') else page
            crop.to_image(resolution=120).original.convert('RGB').save(image_path)
            text = crop.extract_text() or ''
        # Only this historical question/table crop enters the encoder: no web
        # sidebar, label, election result, generated chart, or current news.
        txt_path = root / 'data/multimodal/text' / (row['id'] + '.txt')
        txt_path.parent.mkdir(exist_ok=True, parents=True)
        txt_path.write_text(text)
        # CLIP supports 77 tokens. Disjoint text chunks preserve more than a
        # truncated first line; average their normalized embeddings.
        words = text.split()
        chunks = [' '.join(words[j:j+36]) for j in range(0, min(len(words), 288), 36)] or ['']
        with Image.open(image_path) as im:
            pixels = processor(images=im.convert('RGB'), return_tensors='pt')['pixel_values']
        tokenized = processor(text=chunks, return_tensors='pt', padding=True,
                              truncation=True, max_length=77)
        with torch.inference_mode():
            # get_*_features return projected tensors in the pinned runtime.
            vi = model.vision_model(pixel_values=pixels).pooler_output
            image = model.visual_projection(vi)
            tx = model.text_model(**tokenized).pooler_output
            text_vector = model.text_projection(tx)
            text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
            image = image / image.norm(dim=-1, keepdim=True)
        images.append(image[0].numpy())
        average = text_vector.mean(dim=0).numpy()
        texts.append(average / max(np.linalg.norm(average), 1e-12))
        records.append(dict(id=row['id'], pdf_sha256=row['pdf_sha256'],
            image_sha256=digest(image_path), text_sha256=digest(txt_path),
            page=row['page'], crop=row.get('crop'), text_chunks=len(chunks)))
        print(f'Encoded {i+1}/{len(cases)} {row["id"]}', flush=True)
    destination = root / 'data/multimodal/embeddings.npz'
    np.savez_compressed(destination, image=np.array(images), text=np.array(texts),
                        ids=np.array([r['id'] for r in cases]))
    weights = encoder_path / 'pytorch_model.bin'
    info = dict(encoder=ENCODER_ID, revision=ENCODER_REVISION,
        encoder_weights_sha256=digest(weights), encoder_frozen=True,
        cases_sha256=digest(root/'data/multimodal/cases.json'),
        embeddings_sha256=digest(destination), records=records,
        created_at=datetime.now(timezone.utc).isoformat(), new_service_spend_usd=0)
    (root/'data/multimodal/embeddings-manifest.json').write_text(json.dumps(info, indent=2))


def split_indices(cases):
    train = [i for i, r in enumerate(cases) if r['year'] <= 2020]
    valid = [i for i, r in enumerate(cases) if r['year'] == 2022]
    test = [i for i, r in enumerate(cases) if r['year'] == 2024]
    current = [i for i, r in enumerate(cases) if r['year'] == 2026]
    for a, b in ((train, valid), (train, test), (valid, test)):
        if {cases[i]['race'] for i in a} & {cases[i]['race'] for i in b}:
            raise ValueError('Race leakage across splits')
    if not all((train, valid, test)):
        raise ValueError('All three historical partitions are required')
    return train, valid, test, current


def design(cases, embeddings, train):
    """PCA and all scaling fit exclusively on earlier training elections."""
    nums = np.array([numeric(r) for r in cases], dtype=np.float32)
    scaler = StandardScaler().fit(nums[train])
    parts = [np.clip(scaler.transform(nums), -4, 4)]
    saved = dict(numeric_mean=scaler.mean_, numeric_scale=scaler.scale_)
    for modality in ('text', 'image'):
        pca = PCA(n_components=4, svd_solver='full').fit(embeddings[modality][train])
        projected = pca.transform(embeddings[modality])
        scale = np.maximum(projected[train].std(axis=0), 1e-6)
        parts.append(np.clip(projected / scale, -4, 4))
        saved[modality+'_mean'] = pca.mean_
        saved[modality+'_components'] = pca.components_
        saved[modality+'_scale'] = scale
    return np.concatenate(parts, axis=1).astype(np.float32), nums[:, 0], saved


def columns(variant):
    return {'numeric': list(range(4)), 'numeric_text': list(range(8)),
            'numeric_image': list(range(4))+list(range(8, 12)),
            'multimodal': list(range(12))}[variant]


def network(dim):
    import torch
    class ForecastNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(dim, 8), torch.nn.Tanh(),
                                              torch.nn.Linear(8, 1))

        def forward(self, x, poll):
            # Learn at most a six-point correction to the two-party topline.
            return poll + .6 * torch.tanh(self.layers(x).squeeze(-1))
    return ForecastNetwork()


def aggregate(cases, indices, predictions):
    groups = {}
    for i, value in zip(indices, predictions):
        row = cases[i]
        groups.setdefault(row['race'], []).append((row, float(value)))
    output = []
    for race, observations in sorted(groups.items()):
        # Latest eligible publication only. Repeated polls never inflate N.
        latest = max(row['published_date'] for row, _ in observations)
        latest_obs = [(row, value) for row, value in observations if row['published_date'] == latest]
        margin = float(np.mean([value for _, value in latest_obs])) * 10
        actual = observations[0][0]['actual_dem_margin']
        output.append(dict(race=race, year=observations[0][0]['year'],
            latest_publication=latest, predicted_dem_margin=margin,
            actual_dem_margin=actual, q_democrat=float(ndtr(margin / 6)),
            cases=[row['id'] for row, _ in latest_obs]))
    return output


def metrics(rows):
    y = np.array([r['actual_dem_margin'] > 0 for r in rows], dtype=float)
    q = np.array([r['q_democrat'] for r in rows])
    margin = np.array([r['predicted_dem_margin'] for r in rows])
    actual = np.array([r['actual_dem_margin'] for r in rows])
    return dict(races=len(rows), brier=float(np.mean((q-y)**2)),
        log_loss=float(-np.mean(y*np.log(np.clip(q, 1e-6, 1))+(1-y)*np.log(np.clip(1-q, 1e-6, 1)))),
        accuracy=float(np.mean((q >= .5) == y)),
        margin_rmse=float(np.sqrt(np.mean((margin-actual)**2))))


def train(root):
    import torch
    import safetensors.torch
    root = Path(root)
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cases = read_cases(root)
    emb_path = root/'data/multimodal/embeddings.npz'
    manifest = json.loads((root/'data/multimodal/embeddings-manifest.json').read_text())
    if digest(emb_path) != manifest['embeddings_sha256'] or digest(root/'data/multimodal/cases.json') != manifest['cases_sha256']:
        raise ValueError('Dataset changed; re-encode reviewed sources')
    with np.load(emb_path, allow_pickle=False) as blob:
        if list(blob['ids']) != [r['id'] for r in cases]:
            raise ValueError('Embeddings case order mismatch')
        embeddings = {k: blob[k] for k in ('image', 'text')}
    tr, va, te, cur = split_indices(cases)
    x, poll, preprocessing = design(cases, embeddings, tr)
    artifact = root/'models/multimodal'
    artifact.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(artifact/'preprocessing.npz', **preprocessing)
    targets = np.array([r.get('actual_dem_margin', 0) or 0 for r in cases], dtype=np.float32)/10
    counts = Counter(cases[i]['race'] for i in tr)
    weights = np.array([1/counts[cases[i]['race']] for i in tr], dtype=np.float32)
    weights = torch.tensor(weights/weights.sum())
    all_poll = torch.tensor(poll)
    target = torch.tensor(targets)
    baseline_va = aggregate(cases, va, poll[va])
    baseline_te = aggregate(cases, te, poll[te])
    report = dict(status='EXPERIMENTAL — NOT PROMOTED TO COMPETITION MODEL',
        new_service_spend_usd=0, device='cpu', torch_version=torch.__version__,
        encoder=ENCODER_ID, encoder_revision=ENCODER_REVISION,
        encoder_frozen=True, encoder_parameters=151277313,
        train_years=sorted({cases[i]['year'] for i in tr}),
        validation_years=[2022], test_years=[2024],
        training_cases=len(tr), training_races=len(counts),
        validation_cases=len(va), test_cases=len(te),
        seeds=list(SEEDS), epoch_grid=list(EPOCHS),
        probability_sigma_points=6, correction_limit_points=6,
        cases_sha256=digest(root/'data/multimodal/cases.json'),
        embeddings_sha256=digest(emb_path),
        raw_poll=dict(validation=metrics(baseline_va), test=metrics(baseline_te),
                      validation_predictions=baseline_va, test_predictions=baseline_te), models={})
    for variant in VARIANTS:
        xx = torch.tensor(x[:, columns(variant)])
        checkpoints = {epoch: [] for epoch in EPOCHS}
        traces = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = network(xx.shape[1])
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.03)
            for epoch in range(1, max(EPOCHS)+1):
                model.train()
                pred = model(xx[tr], all_poll[tr])
                loss = ((pred-target[tr]).square()*weights).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch in EPOCHS or epoch == 1:
                    traces.append(dict(seed=seed, epoch=epoch, weighted_margin_mse=float(loss.detach())*100))
                if epoch in EPOCHS:
                    checkpoints[epoch].append({k: v.detach().clone() for k, v in model.state_dict().items()})
            delta = sum(float((v-initial[k]).square().sum()) for k, v in model.state_dict().items())**.5
            if delta <= 0:
                raise ValueError('No actual parameter update occurred')
        validation_grid = []
        for epoch, states in checkpoints.items():
            preds = []
            for state in states:
                model.load_state_dict(state)
                model.eval()
                with torch.inference_mode():
                    preds.append(model(xx, all_poll).numpy())
            ensemble = np.mean(preds, axis=0)
            val = metrics(aggregate(cases, va, ensemble[va]))
            validation_grid.append(dict(epoch=epoch, metrics=val))
        best_epoch = min(validation_grid, key=lambda r: (r['metrics']['margin_rmse'], r['epoch']))['epoch']
        predictions = []
        for seed, state in zip(SEEDS, checkpoints[best_epoch]):
            model.load_state_dict(state)
            model.eval()
            safetensors.torch.save_file(state, str(artifact/f'{variant}-seed{seed}.safetensors'))
            with torch.inference_mode():
                predictions.append(model(xx, all_poll).numpy())
        ensemble = np.mean(predictions, axis=0)
        validation = aggregate(cases, va, ensemble[va])
        test = aggregate(cases, te, ensemble[te])
        item = dict(trainable_parameters=sum(p.numel() for p in model.parameters()),
            selected_epoch=best_epoch, selection='2022 race-level margin RMSE only',
            validation_grid=validation_grid, training_trace=traces,
            parameter_l2_change_last_seed_at_max_epoch=delta, validation=metrics(validation),
            test=metrics(test), validation_predictions=validation, test_predictions=test)
        if cur:
            item['current_research_predictions'] = [dict(id=cases[i]['id'],
                predicted_dem_margin=float(ensemble[i]*10), q_democrat=float(ndtr(ensemble[i]*10/6))) for i in cur]
        if variant == 'multimodal':
            sensitivity = {}
            for name, sl in [('image', slice(8, 12)), ('text', slice(4, 8))]:
                masked = xx.clone()
                masked[:, sl] = 0
                altered = []
                for state in checkpoints[best_epoch]:
                    model.load_state_dict(state)
                    with torch.inference_mode():
                        altered.append(model(masked, all_poll).numpy())
                diff = (ensemble - np.mean(altered, axis=0))*10
                sensitivity[name+'_masked_mean_abs_margin_change'] = float(np.mean(np.abs(diff[te])))
            item['modality_sensitivity'] = sensitivity
        report['models'][variant] = item
        print(variant, json.dumps(item['test']), flush=True)
    report['runtime_seconds'] = round(time.monotonic()-start, 3)
    mm = report['models']['multimodal']['test']
    numeric_test = report['models']['numeric']['test']
    report['promotion'] = dict(promoted=False, minimum_holdout_races=20,
        reasons=['The holdout is smaller than 20 independent race outcomes.',
                 '2026 has a new publisher/layout; broad calibration and tail validation are unfinished.',
                 'On this holdout, the multimodal network is worse than the matched numeric-only network on Brier score and margin RMSE.'],
        multimodal_minus_numeric_brier=mm['brier']-numeric_test['brier'],
        multimodal_minus_numeric_rmse=mm['margin_rmse']-numeric_test['margin_rmse'])
    report['artifacts'] = {p.name: digest(p) for p in sorted(artifact.iterdir()) if p.is_file() and p.suffix in ('.safetensors', '.npz')}
    (root/'reports/multimodal.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (artifact/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def predict(root, case_id):
    """Use saved weights and preprocessing; never fit on a current document."""
    import torch
    from safetensors.torch import load_file
    root = Path(root)
    cases = read_cases(root)
    ids = [r['id'] for r in cases]
    if case_id not in ids:
        raise ValueError('Unknown reviewed case')
    index = ids.index(case_id)
    run = json.loads((root/'models/multimodal/run.json').read_text())
    if digest(root/'data/multimodal/cases.json') != run['cases_sha256']:
        raise ValueError('Cases differ from the trained experiment')
    if digest(root/'data/multimodal/embeddings.npz') != run['embeddings_sha256']:
        raise ValueError('Embeddings differ from the trained experiment')
    artifact = root/'models/multimodal'
    for name, expected in run['artifacts'].items():
        if digest(artifact/name) != expected:
            raise ValueError('Model artifact hash mismatch: ' + name)
    with np.load(root/'data/multimodal/embeddings.npz', allow_pickle=False) as emb, np.load(artifact/'preprocessing.npz', allow_pickle=False) as pre:
        if list(emb['ids']) != ids:
            raise ValueError('Embedding case order mismatch')
        values = np.array(numeric(cases[index]))
        parts = [np.clip((values-pre['numeric_mean'])/pre['numeric_scale'], -4, 4)]
        for modality in ('text', 'image'):
            projected = (emb[modality][index]-pre[modality+'_mean']) @ pre[modality+'_components'].T
            parts.append(np.clip(projected/pre[modality+'_scale'], -4, 4))
        xx = torch.tensor(np.concatenate(parts)[None, :].astype(np.float32))
    model = network(12).eval()
    predictions = []
    for seed in run['seeds']:
        model.load_state_dict(load_file(str(artifact/f'multimodal-seed{seed}.safetensors')))
        with torch.inference_mode():
            predictions.append(float(model(xx, torch.tensor([values[0]], dtype=torch.float32))[0])*10)
    margin = float(np.mean(predictions))
    return dict(case_id=case_id, status=run['status'], predicted_dem_margin=margin,
                q_democrat_conditional_on_major_parties=float(ndtr(margin/6)),
                new_service_spend_usd=0, promoted=False)
