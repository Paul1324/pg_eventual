from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "cache" / "all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@functools.lru_cache(maxsize=1)
def _models():
    from sentence_transformers import SentenceTransformer

    full = SentenceTransformer(str(MODEL_DIR), device="cpu")
    names = [type(m).__name__ for m in full]
    assert names[:2] == ["Transformer", "Pooling"], names
    unnormalized = SentenceTransformer(modules=[full[0], full[1]], device="cpu")
    return full, unnormalized


def embed(texts: list[str], normalize: bool) -> np.ndarray:
    full, unnormalized = _models()
    model = full if normalize else unnormalized
    return np.asarray(model.encode(texts, convert_to_numpy=True, batch_size=16), dtype=np.float32)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


if __name__ == "__main__":
    vectors = embed(["Hello world"], normalize=True)
    print(vectors.shape, vectors[0, :4], float(np.linalg.norm(vectors[0])))
