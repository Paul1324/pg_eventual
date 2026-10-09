from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "all-MiniLM-L6-v2"
BUILD = HERE / "build"
TEMPLATES = HERE / "models"


def build_pooled_onnx(destination: Path) -> None:
    model = onnx.load(str(CACHE / "onnx" / "model.onnx"))
    graph = model.graph
    hidden = "last_hidden_state"
    graph.input.append(helper.make_tensor_value_info("NORMALIZE", TensorProto.BOOL, [1]))
    graph.initializer.extend([
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ev_axis1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "ev_axis2"),
        numpy_helper.from_array(np.array(1e-9, dtype=np.float32), "ev_min_count"),
        numpy_helper.from_array(np.array(1e-12, dtype=np.float32), "ev_min_norm"),
    ])
    graph.node.extend([
        helper.make_node("Cast", ["attention_mask"], ["ev_mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ev_mask_f", "ev_axis2"], ["ev_mask3"]),
        helper.make_node("Mul", [hidden, "ev_mask3"], ["ev_masked"]),
        helper.make_node("ReduceSum", ["ev_masked", "ev_axis1"], ["ev_sum"], keepdims=0),
        helper.make_node("ReduceSum", ["ev_mask3", "ev_axis1"], ["ev_count"], keepdims=0),
        helper.make_node("Max", ["ev_count", "ev_min_count"], ["ev_count_c"]),
        helper.make_node("Div", ["ev_sum", "ev_count_c"], ["ev_mean"]),
        helper.make_node("Mul", ["ev_mean", "ev_mean"], ["ev_sq"]),
        helper.make_node("ReduceSum", ["ev_sq", "ev_axis1"], ["ev_sqsum"], keepdims=1),
        helper.make_node("Sqrt", ["ev_sqsum"], ["ev_norm"]),
        helper.make_node("Max", ["ev_norm", "ev_min_norm"], ["ev_norm_c"]),
        helper.make_node("Div", ["ev_mean", "ev_norm_c"], ["ev_unit"]),
        helper.make_node("Where", ["NORMALIZE", "ev_unit", "ev_mean"], ["EMBEDDING"]),
    ])
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("EMBEDDING", TensorProto.FLOAT, ["batch_size", 384]))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))


def check_pooled_onnx(path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping local graph check")
        return
    import sys

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(TEMPLATES / "eventual_wordpiece" / "1"))
    import reference
    from wordpiece import WordPieceTokenizer

    texts = ["Hello world", "Mean pooling must ignore padding tokens in longer batches, too."]
    ids, mask, types = WordPieceTokenizer(CACHE / "vocab.txt").batch(texts)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for normalize in (True, False):
        (out,) = session.run(["EMBEDDING"], {"input_ids": ids, "attention_mask": mask, "token_type_ids": types,
                                             "NORMALIZE": np.array([normalize])})
        ref = reference.embed(texts, normalize)
        print(f"local ONNX check normalize={normalize}: max|diff| = {np.abs(out - ref).max():.2e}")
        assert np.abs(out - ref).max() < 1e-4


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    pooled = BUILD / "eventual_minilm_pooled" / "1" / "model.onnx"
    build_pooled_onnx(pooled)
    shutil.copy(TEMPLATES / "eventual_minilm_pooled" / "config.pbtxt", pooled.parent.parent / "config.pbtxt")

    tokenizer = BUILD / "eventual_wordpiece"
    shutil.copytree(TEMPLATES / "eventual_wordpiece", tokenizer)
    shutil.copy(CACHE / "vocab.txt", tokenizer / "1" / "vocab.txt")

    ensemble = BUILD / "eventual_text_embedder"
    shutil.copytree(TEMPLATES / "eventual_text_embedder", ensemble)
    (ensemble / "1").mkdir(exist_ok=True)

    for path in BUILD.rglob("__pycache__"):
        shutil.rmtree(path)
    check_pooled_onnx(pooled)
    print(f"built models in {BUILD}")


if __name__ == "__main__":
    main()
