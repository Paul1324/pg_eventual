#!/usr/bin/env bash
set -euo pipefail
REPO=sentence-transformers/all-MiniLM-L6-v2
REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dest="$here/cache/all-MiniLM-L6-v2"
mkdir -p "$dest/onnx" "$dest/1_Pooling"
files=(onnx/model.onnx vocab.txt tokenizer.json tokenizer_config.json special_tokens_map.json config.json
       sentence_bert_config.json modules.json config_sentence_transformers.json 1_Pooling/config.json model.safetensors)
for f in "${files[@]}"; do
  if [ ! -s "$dest/$f" ]; then
    curl --fail --location --silent --show-error --retry 3 -o "$dest/$f" \
      "https://huggingface.co/$REPO/resolve/$REVISION/$f"
  fi
done
echo "$REVISION" > "$dest/REVISION"
sha256sum "$dest/onnx/model.onnx" "$dest/model.safetensors" "$dest/vocab.txt"
