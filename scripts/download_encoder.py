"""One-time free download. No inference API, token, GPU rental, or paid fallback."""
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', type=Path, required=True)
    args = ap.parse_args()
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import snapshot_download
    path = snapshot_download('openai/clip-vit-base-patch32',
        revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        cache_dir=str(args.cache_dir), token=False,
        allow_patterns=['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                        'vocab.json', 'merges.txt', 'special_tokens_map.json', 'pytorch_model.bin'])
    print(path)
