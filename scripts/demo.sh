#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
{
  echo "pg_eventual demo ($(date '+%Y-%m-%d %H:%M'))"
  bash scripts/cluster.sh start
  for _ in $(seq 1 60); do
    curl --fail --silent --max-time 2 "$TRITON_URL/v2/health/ready" >/dev/null && break
    sleep 1
  done
  if ! curl --fail --silent --max-time 3 "$TRITON_URL/v2/models/eventual_text_embedder/versions/1/ready" >/dev/null; then
    echo "Deploying the embedding models to Triton..."
    bash serving/deploy.sh
  fi
  psql -X -q -f demo/demo.sql
  echo
  echo "=== 9. Independent check: Triton pipeline vs sentence-transformers on CPU ========="
  HF_HUB_OFFLINE=1 .venv-ref/bin/python serving/compare.py 2>/dev/null
} 2>&1 | sed -u "s|$PROJECT_ROOT/||g" | tee demo/output.txt
