#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
profile_flag=()
if [ "${1:-}" = "--release" ]; then profile_flag=(--release); fi
cargo pgrx install --pg-config "$PG_CONFIG" "${profile_flag[@]}"
bash scripts/cluster.sh restart
