#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
root="$PROJECT_ROOT/.state/pg"
data="$root/data"
log="$root/server.log"
umask 077

init() {
  mkdir -p "$root" "$PGHOST"
  if [ ! -f "$data/PG_VERSION" ]; then
    initdb --pgdata="$data" --username="$PGUSER" --auth-local=trust --auth-host=reject \
      --encoding=UTF8 --locale=C.UTF-8 >/dev/null
    cat >>"$data/postgresql.conf" <<EOF

# --- pg_eventual development settings ---
listen_addresses = ''
port = $PGPORT
unix_socket_directories = '$PGHOST'
shared_preload_libraries = 'pg_eventual'
max_worker_processes = 16
eventual.database = 'eventual_dev, eventual_dev2'
eventual.workers = 2
eventual.poll_interval_ms = 1000
eventual.batch_size = 32
eventual.retry_base_ms = 200
eventual.triton_url = '$TRITON_URL'
log_min_messages = info
log_line_prefix = '%m [%p] %b '
EOF
    echo "initialized $data"
  fi
}

running() { pg_ctl --pgdata="$data" status >/dev/null 2>&1; }

start() {
  init
  if ! running; then
    pg_ctl --pgdata="$data" --log="$log" --wait --timeout=60 start >/dev/null
  fi
  for db in eventual_dev eventual_dev2; do
    psql -X -q -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$db'" | grep -q 1 ||
      psql -X -q -d postgres -c "CREATE DATABASE $db" >/dev/null
  done
  psql -X -tAc "SELECT 'PostgreSQL ' || current_setting('server_version') || ' on ' || current_setting('unix_socket_directories')"
}

stop() {
  if [ -f "$data/PG_VERSION" ] && running; then
    pg_ctl --pgdata="$data" --mode=fast --wait stop >/dev/null
  fi
}

case "${1:-status}" in
  init) init ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) pg_ctl --pgdata="$data" status ;;
  logs) tail -n "${2:-80}" "$log" ;;
  psql) shift; exec psql -X "$@" ;;
  destroy) stop; rm -rf "$root"; echo "removed $root" ;;
  *) echo "usage: scripts/cluster.sh {init|start|stop|restart|status|logs [n]|psql|destroy}" >&2; exit 2 ;;
esac
