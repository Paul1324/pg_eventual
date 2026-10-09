import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference  # noqa: E402

URL = os.environ.get("TRITON_URL", "http://localhost:8000")
TEXTS = [
    "Hello world",
    "PostgreSQL background workers compute derived columns asynchronously.",
    "Café naïve résumé \u2014 accents, dashes and \"quotes\"!",
    "東京は日本の首都です。中文字符也应该被处理。",
    "Emoji 🦀🐘 and symbols ±§©®™ plus numbers 3.14159 and 1,000,000.",
    "UPPER lower MiXeD case, tabs\tand\nnewlines.",
    " ".join(["long"] * 400),
    "a",
]


def triton_embed(texts, normalize):
    body = {"inputs": [{"name": "TEXT", "shape": [len(texts)], "datatype": "BYTES", "data": texts},
                       {"name": "NORMALIZE", "shape": [1], "datatype": "BOOL", "data": [normalize]}]}
    request = urllib.request.Request(f"{URL}/v2/models/eventual_text_embedder/versions/1/infer",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        out = json.load(response)["outputs"][0]
    return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "models" / "eventual_wordpiece" / "1"))
    from wordpiece import WordPieceTokenizer  # noqa: E402

    ours = WordPieceTokenizer(reference.MODEL_DIR / "vocab.txt", max_length=256)
    hf = reference.tokenizer()
    for text in TEXTS:
        expected = hf(text, truncation=True, max_length=256)["input_ids"]
        assert ours.encode(text) == expected, f"tokenizer mismatch for {text[:40]!r}"
    print(f"tokenizer: {len(TEXTS)} texts match Hugging Face ids")

    worst = {}
    for normalize in (True, False):
        got = triton_embed(TEXTS, normalize)
        ref = reference.embed(TEXTS, normalize)
        diff = np.abs(got - ref).max(axis=1)
        cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
        worst[normalize] = (float(diff.max()), float(cos.min()))
        print(f"normalize={normalize}: max|diff|={diff.max():.2e} min cosine={cos.min():.8f} "
              f"norms={np.linalg.norm(got, axis=1).round(4).tolist()}")
    return worst


if __name__ == "__main__":
    main()
