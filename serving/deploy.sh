#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="${TRITON_MODEL_REPOSITORY:-}"
if [ -z "$repo" ] || [ ! -d "$repo" ]; then
  echo "set TRITON_MODEL_REPOSITORY to the host directory your Triton server loads models from" >&2
  echo "(export it, or put it in scripts/env.local.sh)" >&2
  exit 2
fi
url="${TRITON_URL:-http://localhost:8000}"
build="$here/build"
models=(eventual_wordpiece eventual_minilm_pooled eventual_text_embedder)
legacy=(eventual_minilm_onnx)

api() {
  local code
  code=$(curl --silent --show-error --output /tmp/eventual_triton_api.out --write-out '%{http_code}' \
    --max-time 300 -X POST "$url/v2/repository/models/$2/$1")
  if [ "$code" != "200" ]; then
    echo "$1 $2 failed (HTTP $code): $(cat /tmp/eventual_triton_api.out)" >&2
    return 1
  fi
}

remove() {
  for model in eventual_text_embedder eventual_minilm_pooled eventual_wordpiece "${legacy[@]}"; do
    case "$model" in eventual_*) ;; *) echo "refusing to touch $model" >&2; exit 1 ;; esac
    api unload "$model" 2>/dev/null || true
    rm -rf "${repo:?}/$model"
  done
}

if [ "${1:-}" = "remove" ]; then
  remove
  echo "removed eventual_* models"
  exit 0
fi

if [ ! -s "$build/eventual_minilm_pooled/1/model.onnx" ]; then
  [ -s "$here/cache/all-MiniLM-L6-v2/onnx/model.onnx" ] || bash "$here/fetch_model.sh"
  "$here/../.venv-ref/bin/python" "$here/build_models.py"
fi
remove
for model in "${models[@]}"; do
  cp -r "$build/$model" "$repo/$model"
  chmod -R a+rX "$repo/$model"
done
for model in "${models[@]}"; do
  api load "$model"
done
curl --silent -X POST "$url/v2/repository/index" | python3 -c '
import json, sys
for m in json.load(sys.stdin):
    print("%-28s v%s %s %s" % (m["name"], m.get("version", "?"), m.get("state", ""), m.get("reason", "")))'
