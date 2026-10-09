#!/usr/bin/env bash
export PATH="$HOME/.pgrx/18.6/pgrx-install/bin:$HOME/.local/bin:$PATH"
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck source=/dev/null
  . "$HOME/.cargo/env"
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PG_CONFIG="$HOME/.pgrx/18.6/pgrx-install/bin/pg_config"
export PGHOST="$PROJECT_ROOT/.state/pg/socket"
export PGPORT=55450
export PGUSER="${USER:-$(id -un)}"
export PGDATABASE=eventual_dev
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
if [ -f "$PROJECT_ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$PROJECT_ROOT/scripts/env.local.sh"
fi
export TRITON_URL="${TRITON_URL:-http://localhost:8000}"
