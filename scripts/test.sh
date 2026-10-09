#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
what=all
case "${1:-}" in
  all|unit|integration) what="$1"; shift ;;
  ""|-*) ;;
  *) echo "usage: scripts/test.sh [all|unit|integration] [pytest args]" >&2; exit 2 ;;
esac
if [ "$what" = "all" ] || [ "$what" = "unit" ]; then
  cargo test --lib --quiet
fi
if [ "$what" = "all" ] || [ "$what" = "integration" ]; then
  python=.venv-ref/bin/python
  if [ ! -x "$python" ]; then
    python=.venv/bin/python
    if [ ! -x "$python" ]; then
      python3 -m venv .venv
      .venv/bin/pip install --quiet -r tests/requirements.txt
    fi
  fi
  bash scripts/cluster.sh start >/dev/null
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$python" -m pytest tests/integration "$@"
fi
