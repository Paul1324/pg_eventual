use crate::guc;
use crate::processors::ColumnSpec;
use crate::security::as_role;
use crate::trigger::WRITEBACK;
use crate::triton;
use pgrx::bgworkers::{BackgroundWorker, SignalWakeFlags};
use pgrx::pg_sys::panic::CaughtError;
use pgrx::pg_sys::{self, PgTryBuilder};
use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn shutdown_requested() -> bool {
    unsafe { (&raw const pg_sys::ShutdownRequestPending).read_volatile() != 0 }
}

fn describe(error: CaughtError) -> String {
    match error {
        CaughtError::PostgresError(report) | CaughtError::ErrorReport(report) => match report.detail() {
            Some(detail) if !detail.is_empty() => format!("{} ({detail})", report.message()),
            _ => report.message().to_string(),
        },
        CaughtError::RustPanic { ereport, .. } => ereport.message().to_string(),
    }
}

fn txn<R>(body: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
    unsafe {
        pg_sys::SetCurrentStatementStartTimestamp();
        pg_sys::StartTransactionCommand();
        pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
    }
    let outcome = PgTryBuilder::new(AssertUnwindSafe(|| {
        let result = body();
        if result.is_ok() {
            unsafe {
                pg_sys::PopActiveSnapshot();
                pg_sys::CommitTransactionCommand();
            }
        }
        result
    }))
    .catch_others(|error| Err(describe(error)))
    .execute();
    if outcome.is_err() {
        WRITEBACK.store(false, Ordering::SeqCst);
        unsafe { pg_sys::AbortCurrentTransaction() };
    }
    outcome
}

// Signals would interrupt blocking socket calls with EINTR; block them during remote requests.
struct SignalsBlocked;

impl SignalsBlocked {
    fn new() -> Self {
        unsafe { pg_sys::BackgroundWorkerBlockSignals() };
        SignalsBlocked
    }
}

impl Drop for SignalsBlocked {
    fn drop(&mut self) {
        unsafe { pg_sys::BackgroundWorkerUnblockSignals() };
    }
}

fn spi_err(e: pgrx::spi::SpiError) -> String {
    e.to_string()
}

struct Claimed {
    spec: Rc<ColumnSpec>,
    row_key: Vec<String>,
    generation: i64,
    attempts: i32,
    inputs: Option<Vec<Value>>,
}

enum Outcome {
    Done,
    Stale,
    Gone,
    Waiting,
}

struct Worker {
    idle_reason: &'static str,
    index: i32,
    processed: i64,
    failed: i64,
    last_error: Option<String>,
    reported_state: Option<String>,
    last_heartbeat: Option<Instant>,
    last_recovery: Option<Instant>,
    ready: bool,
    last_ready_check: Option<Instant>,
}

#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn eventual_worker_main(arg: pg_sys::Datum) {
    let slot = unsafe { i32::from_datum(arg, false) }.unwrap_or(0);
    let index = slot % guc::WORKERS.get().max(1);
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    let database = match BackgroundWorker::get_extra() {
        extra if !extra.is_empty() => extra.to_string(),
        _ => guc::databases().into_iter().next().unwrap_or_else(|| "postgres".into()),
    };
    let role = guc::string_setting(&guc::ROLE).filter(|r| !r.is_empty());
    BackgroundWorker::connect_worker_to_spi(Some(&database), role.as_deref());
    crate::queue::register_worker(slot, unsafe { pg_sys::MyProcNumber }, unsafe { pg_sys::MyDatabaseId });
    log!("pg_eventual worker {index} started in database \"{database}\"");

    let mut worker = Worker {
        idle_reason: "",
        index,
        processed: 0,
        failed: 0,
        last_error: None,
        reported_state: None,
        last_heartbeat: None,
        last_recovery: None,
        ready: false,
        last_ready_check: None,
    };
    while !shutdown_requested() {
        if BackgroundWorker::sighup_received() {
            unsafe { pg_sys::ProcessConfigFile(pg_sys::GucContext::PGC_SIGHUP) };
        }
        if let Some(idle) = worker.iteration() {
            if !BackgroundWorker::wait_latch(Some(idle)) {
                break;
            }
        }
    }
    worker.shutdown();
    crate::queue::register_worker(slot, -1, pg_sys::InvalidOid);
    log!("pg_eventual worker {index} exiting");
    // Exit code 0 would unregister this static worker; 1 lets the postmaster restart it.
    unsafe { pg_sys::proc_exit(1) };
}

impl Worker {
    fn iteration(&mut self) -> Option<Duration> {
        let poll = Duration::from_millis(guc::POLL_INTERVAL_MS.get() as u64);
        let result = match self.step() {
            true => None,
            false if !self.ready => Some(Duration::from_secs(2)),
            false if self.idle_reason == "paused" => Some(poll),
            false => Some(match txn(next_due) {
                Ok(Some(due)) => poll.min(due.max(Duration::from_millis(5))),
                _ => poll,
            }),
        };
        result
    }

    fn step(&mut self) -> bool {
        if !self.ready || self.last_ready_check.map_or(true, |t| t.elapsed() > Duration::from_secs(10)) {
            self.ready = txn(|| {
                Spi::get_one::<bool>("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_eventual')")
                    .map_err(spi_err)
                    .map(|v| v.unwrap_or(false))
            })
            .unwrap_or(false);
            self.last_ready_check = Some(Instant::now());
            if !self.ready {
                self.idle_reason = "not ready";
                return false;
            }
        }

        let paused = match txn(|| self.heartbeat()) {
            Ok(paused) => paused,
            Err(error) => {
                warning!("pg_eventual worker {}: {error}", self.index);
                self.ready = false;
                return false;
            }
        };
        if paused {
            self.idle_reason = "paused";
            return false;
        }

        if self.last_recovery.map_or(true, |t| t.elapsed() > Duration::from_secs(30)) {
            if let Err(error) = txn(recover_orphans) {
                warning!("pg_eventual worker {}: recovering orphaned cells failed: {error}", self.index);
            }
            self.last_recovery = Some(Instant::now());
        }

        let batch = match txn(|| claim(guc::BATCH_SIZE.get())) {
            Ok(batch) => batch,
            Err(error) => {
                warning!("pg_eventual worker {}: claiming work failed: {error}", self.index);
                return false;
            }
        };
        if batch.is_empty() {
            self.idle_reason = "nothing claimed";
            return false;
        }
        let mut groups: Vec<Vec<Claimed>> = Vec::new();
        for cell in batch {
            match groups.iter_mut().find(|g| g[0].spec.column_id == cell.spec.column_id) {
                Some(group) => group.push(cell),
                None => groups.push(vec![cell]),
            }
        }
        let mut groups = groups.into_iter();
        for group in groups.by_ref() {
            self.process_group(group);
            if shutdown_requested() {
                break;
            }
        }
        let leftover: Vec<Claimed> = groups.flatten().collect();
        if !leftover.is_empty() {
            let _ = txn(|| release(&leftover));
        }
        true
    }

    fn process_group(&mut self, group: Vec<Claimed>) {
        let spec = group[0].spec.clone();
        match spec.kind.as_str() {
            "sql" if group.len() > 1 => {
                let started = Instant::now();
                match txn(|| write_batch(&group, started, BatchPayload::Sql)) {
                    Ok(done) => self.processed += done,
                    Err(error) => {
                        log!("pg_eventual worker {}: batch of {} cells fell back to single cells: {error}", self.index, group.len());
                        group.into_iter().for_each(|cell| self.process(cell))
                    }
                }
            }
            "triton" | "http" if group.len() > 1 => {
                let max_batch = match spec.kind.as_str() {
                    "triton" => triton::TritonConfig::from_json(&spec.config).ok().and_then(|c| c.max_batch),
                    _ => crate::http::HttpConfig::from_json(&spec.config).ok().and_then(|c| c.max_batch),
                };
                match max_batch {
                    Some(size) if size > 1 => {
                        for chunk in group.chunks(size) {
                            self.process_remote_batch(chunk);
                            if shutdown_requested() {
                                break;
                            }
                        }
                    }
                    _ => group.into_iter().for_each(|cell| self.process(cell)),
                }
            }
            _ => group.into_iter().for_each(|cell| self.process(cell)),
        }
    }

    fn process_remote_batch(&mut self, cells: &[Claimed]) {
        let started = Instant::now();
        match run_remote_batch(cells) {
            Ok(results) => {
                let values: Vec<Value> = results.iter().map(|(v, _)| v.clone()).collect();
                let mut meta = results[0].1.clone();
                meta.as_object_mut().map(|m| m.remove("attempt"));
                match txn(|| write_batch(cells, started, BatchPayload::Json { values: &values, meta: &meta })) {
                    Ok(done) => self.processed += done,
                    Err(error) => {
                        log!("pg_eventual worker {}: batch write of {} cells fell back to single cells: {error}", self.index, cells.len());
                        for (cell, (value, meta)) in cells.iter().zip(results) {
                            match txn(|| write_json(cell, &value, meta, started)) {
                                Ok(Outcome::Done) => self.processed += 1,
                                Ok(_) => {}
                                Err(error) => self.fail(cell, &error, started),
                            }
                        }
                    }
                }
            }
            Err(error) => cells.iter().for_each(|cell| self.fail(cell, &error, started)),
        }
    }

    fn fail(&mut self, cell: &Claimed, error: &str, started: Instant) {
        self.failed += 1;
        self.last_error = Some(error.to_string());
        if let Err(record_error) = txn(|| record_failure(cell, error, started)) {
            warning!("pg_eventual worker {}: could not record failure ({error}): {record_error}", self.index);
        }
    }

    fn heartbeat(&mut self) -> Result<bool, String> {
        let paused = Spi::get_one::<bool>("SELECT paused FROM eventual.control")
            .map_err(spi_err)?
            .unwrap_or(false);
        let state = if paused { "paused" } else { "running" };
        let due = self.last_heartbeat.map_or(true, |t| t.elapsed() > Duration::from_secs(2));
        if due || self.reported_state.as_deref() != Some(state) {
            Spi::run_with_args(
                "INSERT INTO eventual.workers AS w (worker_index, pid, state, started_at, heartbeat_at, processed, failed, last_error)
                 VALUES ($1, pg_backend_pid(), $2, now(), now(), $3, $4, $5)
                 ON CONFLICT (worker_index) DO UPDATE
                 SET pid = EXCLUDED.pid, state = EXCLUDED.state, heartbeat_at = now(),
                     processed = EXCLUDED.processed, failed = EXCLUDED.failed, last_error = EXCLUDED.last_error,
                     started_at = CASE WHEN w.pid IS DISTINCT FROM EXCLUDED.pid THEN now() ELSE w.started_at END",
                &[
                    self.index.into(),
                    state.into(),
                    self.processed.into(),
                    self.failed.into(),
                    self.last_error.clone().into(),
                ],
            )
            .map_err(spi_err)?;
            self.reported_state = Some(state.to_string());
            self.last_heartbeat = Some(Instant::now());
        }
        Ok(paused)
    }

    fn shutdown(&mut self) {
        let index = self.index;
        let _ = txn(|| {
            Spi::run(
                "UPDATE eventual.cells SET state = 'pending', worker_pid = NULL, started_at = NULL,
                        attempts = greatest(attempts - 1, 0)
                 WHERE worker_pid = pg_backend_pid() AND state = 'running'",
            )
            .map_err(spi_err)?;
            Spi::run_with_args(
                "UPDATE eventual.workers SET state = 'stopped', heartbeat_at = now() WHERE worker_index = $1",
                &[index.into()],
            )
            .map_err(spi_err)
        });
    }

    fn process(&mut self, cell: Claimed) {
        let started = Instant::now();
        let result = match cell.spec.kind.as_str() {
            "sql" => txn(|| write_sql(&cell, started)),
            "triton" | "http" => match run_remote(&cell) {
                Ok((value, meta)) => txn(|| write_json(&cell, &value, meta, started)),
                Err(error) => Err(error),
            },
            other => Err(format!("unsupported processor kind \"{other}\"")),
        };
        match result {
            Ok(Outcome::Done) => self.processed += 1,
            Ok(Outcome::Stale | Outcome::Gone | Outcome::Waiting) => {}
            Err(error) => self.fail(&cell, &error, started),
        }
    }
}

const LOCK_CELLS_BATCH: &str = "
SELECT k.ev_i, c.generation, c.state
FROM (SELECT ev_i, ($1::text[])[(ev_i - 1) * $4 + 1 : ev_i * $4] AS ev_key
      FROM generate_series(1, $2) AS ev_i) k
JOIN eventual.cells c ON c.column_id = $3 AND c.row_key = k.ev_key
ORDER BY k.ev_i
FOR UPDATE OF c";

fn cells_batch_sql(set: &str) -> String {
    format!(
        "UPDATE eventual.cells c SET {set}
         FROM (SELECT ev_i, ($1::text[])[(ev_i - 1) * $4 + 1 : ev_i * $4] AS ev_key, ($3::bigint[])[ev_i] AS ev_gen
               FROM generate_series(1, cardinality($3::bigint[])) AS ev_i) k
         WHERE c.column_id = $2 AND c.row_key = k.ev_key AND c.generation = k.ev_gen"
    )
}

fn flatten_keys<'a>(cells: impl Iterator<Item = &'a Claimed>) -> (Vec<String>, Vec<i64>) {
    let mut keys = Vec::new();
    let mut generations = Vec::new();
    for cell in cells {
        keys.extend(cell.row_key.iter().cloned());
        generations.push(cell.generation);
    }
    (keys, generations)
}

enum BatchPayload<'a> {
    Sql,
    Json { values: &'a [Value], meta: &'a Value },
}

fn write_batch(group: &[Claimed], started: Instant, payload: BatchPayload) -> Result<i64, String> {
    let spec = &group[0].spec;
    let key_len = spec.pk_sql.len() as i32;
    let count = group.len() as i32;
    let (keys, _) = flatten_keys(group.iter());
    let update_sql = match payload {
        BatchPayload::Sql => spec.sql_batch_writeback()?,
        BatchPayload::Json { .. } => spec.json_batch_writeback(),
    };
    Spi::connect_mut(|client| {
        let mut locked = std::collections::HashSet::new();
        for row in client.update(&spec.lock_rows_batch_sql(), None, &[keys.clone().into(), count.into()]).map_err(spi_err)? {
            locked.insert(row.get::<i32>(1).map_err(spi_err)?.unwrap_or(0) as usize - 1);
        }
        let mut existing = std::collections::HashSet::new();
        if locked.len() < group.len() {
            for row in client.select(&spec.rows_exist_batch_sql(), None, &[keys.clone().into(), count.into()]).map_err(spi_err)? {
                existing.insert(row.get::<i32>(1).map_err(spi_err)?.unwrap_or(0) as usize - 1);
            }
        }
        let mut current = Vec::new();
        for row in client
            .update(LOCK_CELLS_BATCH, None, &[keys.clone().into(), count.into(), spec.column_id.into(), key_len.into()])
            .map_err(spi_err)?
        {
            let i = row.get::<i32>(1).map_err(spi_err)?.unwrap_or(0) as usize - 1;
            let generation = row.get::<i64>(2).map_err(spi_err)?.unwrap_or(-1);
            let state = row.get::<String>(3).map_err(spi_err)?.unwrap_or_default();
            if state == "running" && generation == group[i].generation && locked.contains(&i) {
                current.push(i);
            }
        }
        let gone: Vec<&Claimed> = (0..group.len())
            .filter(|i| !locked.contains(i) && !existing.contains(i))
            .map(|i| &group[i])
            .collect();
        if !gone.is_empty() {
            let (gone_keys, gone_gens) = flatten_keys(gone.into_iter());
            client
                .update(
                    &format!("DELETE FROM eventual.cells c USING (SELECT ($1::text[])[(ev_i - 1) * $4 + 1 : ev_i * $4] AS ev_key, ($3::bigint[])[ev_i] AS ev_gen FROM generate_series(1, cardinality($3::bigint[])) AS ev_i) k WHERE c.column_id = $2 AND c.row_key = k.ev_key AND c.generation = k.ev_gen"),
                    None,
                    &[gone_keys.into(), spec.column_id.into(), gone_gens.into(), key_len.into()],
                )
                .map_err(spi_err)?;
        }
        let busy: Vec<&Claimed> = (0..group.len()).filter(|i| !locked.contains(i) && existing.contains(i)).map(|i| &group[i]).collect();
        if !busy.is_empty() {
            let (busy_keys, busy_gens) = flatten_keys(busy.into_iter());
            client
                .update(
                    &cells_batch_sql("state = 'pending', worker_pid = NULL, started_at = NULL, attempts = greatest(c.attempts - 1, 0), next_attempt_at = now() + interval '50 milliseconds'"),
                    None,
                    &[busy_keys.into(), spec.column_id.into(), busy_gens.into(), key_len.into()],
                )
                .map_err(spi_err)?;
        }
        if current.is_empty() {
            return Ok(0);
        }
        let (current_keys, _) = flatten_keys(current.iter().map(|&i| &group[i]));
        let third = match payload {
            BatchPayload::Sql => spec.effective_params.clone(),
            BatchPayload::Json { values, .. } => Value::Array(
                current.iter().map(|&i| json!({ spec.target_name.clone(): values[i].clone() })).collect(),
            ),
        };
        let mut updated = std::collections::HashSet::new();
        {
            // Processors compute and write as the table owner, never as the worker's role.
            let _guard = WritebackGuard::new();
            let rows = as_role(spec.owner(), true, || {
                client.update(&update_sql, None, &[current_keys.clone().into(), (current.len() as i32).into(), JsonB(third).into()])
            })
            .map_err(spi_err)?;
            for row in rows {
                updated.insert(current[row.get::<i32>(1).map_err(spi_err)?.unwrap_or(0) as usize - 1]);
            }
        }
        let waiting: Vec<&Claimed> = current.iter().filter(|i| !updated.contains(i)).map(|&i| &group[i]).collect();
        if !waiting.is_empty() {
            let (w_keys, w_gens) = flatten_keys(waiting.into_iter());
            client
                .update(&cells_batch_sql("state = 'waiting', worker_pid = NULL"), None,
                        &[w_keys.into(), spec.column_id.into(), w_gens.into(), key_len.into()])
                .map_err(spi_err)?;
        }
        let done: Vec<usize> = current.iter().copied().filter(|i| updated.contains(i)).collect();
        let (d_keys, d_gens) = flatten_keys(done.iter().map(|&i| &group[i]));
        let meta = match payload {
            BatchPayload::Sql => {
                let mut meta = base_meta(&group[0]);
                meta["batch"] = json!(group.len());
                meta.as_object_mut().map(|m| m.remove("attempt"));
                meta
            }
            BatchPayload::Json { meta, .. } => meta.clone(),
        };
        client
            .update(
                &cells_batch_sql(
                    "state = 'done', finished_at = clock_timestamp(), duration_ms = $5, worker_pid = NULL, last_error = NULL, result_meta = $6",
                ),
                None,
                &[
                    d_keys.into(),
                    spec.column_id.into(),
                    d_gens.into(),
                    key_len.into(),
                    (started.elapsed().as_secs_f64() * 1000.0).into(),
                    JsonB(meta).into(),
                ],
            )
            .map_err(spi_err)?;
        Ok(done.len() as i64)
    })
}

fn run_remote_batch(cells: &[Claimed]) -> Result<Vec<(Value, Value)>, String> {
    let spec = &cells[0].spec;
    if spec.kind == "http" {
        return run_http_batch(cells);
    }
    let cfg = triton::TritonConfig::from_json(&spec.config).map_err(|e| format!("invalid triton config: {e}"))?;
    let rows: Vec<Vec<Value>> = cells
        .iter()
        .map(|c| c.inputs.clone().ok_or_else(|| "inputs were not read".to_string()))
        .collect::<Result<_, _>>()?;
    let request = triton::build_batch_request(&cfg, &rows, &spec.effective_params)?;
    let base_url = cfg
        .url
        .clone()
        .or_else(|| guc::string_setting(&guc::TRITON_URL))
        .ok_or("no Triton URL configured (set eventual.triton_url or the processor's \"url\")")?;
    let timeout = Duration::from_millis(cfg.timeout_ms.unwrap_or(guc::HTTP_TIMEOUT_MS.get() as u64));
    let response = {
        let _blocked = SignalsBlocked::new();
        triton::infer(&base_url, &cfg, &request, timeout)
    }?;
    let results = triton::parse_batch_response(&cfg, &response, cells.len(), spec.result_is_array)?;
    Ok(cells
        .iter()
        .zip(results)
        .map(|(cell, result)| {
            let mut meta = base_meta(cell);
            meta["model_name"] = json!(result.model_name);
            meta["model_version"] = json!(result.model_version);
            meta["batch"] = json!(cells.len());
            (result.value, meta)
        })
        .collect())
}

const CLAIM_SQL: &str = "
WITH claimable AS (
    SELECT column_id, row_key
    FROM eventual.cells
    WHERE state = 'pending' AND next_attempt_at <= now()
    ORDER BY next_attempt_at, enqueued_at
    LIMIT $1
    FOR UPDATE SKIP LOCKED
)
UPDATE eventual.cells c
SET state = 'running', attempts = c.attempts + 1, started_at = clock_timestamp(),
    finished_at = NULL, worker_pid = pg_backend_pid()
FROM claimable
WHERE c.column_id = claimable.column_id AND c.row_key = claimable.row_key
RETURNING c.column_id, c.row_key, c.generation, c.attempts";

fn claim(limit: i32) -> Result<Vec<Claimed>, String> {
    Spi::connect_mut(|client| {
        let mut rows: Vec<(i64, Vec<String>, i64, i32)> = Vec::new();
        for row in client.update(CLAIM_SQL, None, &[limit.into()]).map_err(spi_err)? {
            rows.push((
                row.get::<i64>(1).map_err(spi_err)?.unwrap_or_default(),
                row.get::<Vec<String>>(2).map_err(spi_err)?.unwrap_or_default(),
                row.get::<i64>(3).map_err(spi_err)?.unwrap_or_default(),
                row.get::<i32>(4).map_err(spi_err)?.unwrap_or_default(),
            ));
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
        ids.sort_unstable();
        ids.dedup();
        let mut specs: HashMap<i64, Rc<ColumnSpec>> = HashMap::new();
        for row in client
            .select(
                "SELECT to_jsonb(s) FROM eventual._column_specs s WHERE s.column_id = ANY($1)",
                None,
                &[ids.into()],
            )
            .map_err(spi_err)?
        {
            let JsonB(value) = row.get::<JsonB>(1).map_err(spi_err)?.ok_or("missing column spec")?;
            let mut spec: ColumnSpec =
                serde_json::from_value(value).map_err(|e| format!("invalid column spec: {e}"))?;
            if spec.kind == "http" {
                if let Ok(cfg) = crate::http::HttpConfig::from_json(&spec.config) {
                    let names = cfg.secret_names();
                    if !names.is_empty() {
                        for row in client
                            .select("SELECT name, value FROM eventual.secrets WHERE name = ANY($1)", None, &[names.into()])
                            .map_err(spi_err)?
                        {
                            let name = row.get::<String>(1).map_err(spi_err)?.unwrap_or_default();
                            let value = row.get::<String>(2).map_err(spi_err)?.unwrap_or_default();
                            spec.secrets.insert(name, value);
                        }
                    }
                }
            }
            specs.insert(spec.column_id, Rc::new(spec));
        }

        let mut items = Vec::new();
        for (column_id, row_key, generation, attempts) in rows {
            let Some(spec) = specs.get(&column_id).cloned() else {
                continue;
            };
            let inputs = if spec.kind == "sql" {
                None
            } else {
                let found = as_role(spec.owner(), true, || {
                    client
                        .select(&spec.read_inputs_sql(), Some(1), &[row_key.clone().into()])
                        .map(|t| t.first().get::<JsonB>(1).ok().flatten())
                })
                .map_err(spi_err)?;
                match found {
                    Some(JsonB(Value::Array(values))) => Some(values),
                    _ => {
                        client
                            .update(
                                "DELETE FROM eventual.cells WHERE column_id = $1 AND row_key = $2",
                                None,
                                &[column_id.into(), row_key.clone().into()],
                            )
                            .map_err(spi_err)?;
                        continue;
                    }
                }
            };
            if let Some(values) = &inputs {
                if spec.strict && values.iter().any(Value::is_null) {
                    client
                        .update(
                            "UPDATE eventual.cells SET state = 'waiting', worker_pid = NULL
                             WHERE column_id = $1 AND row_key = $2 AND generation = $3",
                            None,
                            &[column_id.into(), row_key.clone().into(), generation.into()],
                        )
                        .map_err(spi_err)?;
                    continue;
                }
            }
            items.push(Claimed { spec, row_key, generation, attempts, inputs });
        }
        Ok(items)
    })
}

fn next_due() -> Result<Option<Duration>, String> {
    let seconds = Spi::get_one::<f64>(
        "SELECT extract(epoch FROM min(next_attempt_at) - now())::float8 FROM eventual.cells WHERE state = 'pending'",
    )
    .map_err(spi_err)?;
    Ok(seconds.map(|s| Duration::from_secs_f64(s.max(0.0))))
}

fn release(cells: &[Claimed]) -> Result<(), String> {
    for cell in cells {
        Spi::run_with_args(
            "UPDATE eventual.cells SET state = 'pending', worker_pid = NULL, started_at = NULL,
                    attempts = greatest(attempts - 1, 0)
             WHERE column_id = $1 AND row_key = $2 AND generation = $3 AND state = 'running'",
            &[cell.spec.column_id.into(), cell.row_key.clone().into(), cell.generation.into()],
        )
        .map_err(spi_err)?;
    }
    Ok(())
}

fn recover_orphans() -> Result<(), String> {
    Spi::run(
        "UPDATE eventual.cells c
         SET state = CASE WHEN c.attempts >= p.max_attempts THEN 'failed' ELSE 'pending' END,
             last_error = CASE WHEN c.attempts >= p.max_attempts
                               THEN 'worker exited while computing this value' ELSE c.last_error END,
             worker_pid = NULL, next_attempt_at = now()
         FROM eventual.derived_columns dc, eventual.processors p
         WHERE dc.id = c.column_id AND p.id = dc.processor_id
           AND c.state = 'running'
           AND (c.worker_pid IS NULL OR c.worker_pid NOT IN (
                SELECT pid FROM pg_stat_activity WHERE backend_type = 'pg_eventual worker'))",
    )
    .map_err(spi_err)
}

fn lock_for_writeback(client: &mut pgrx::spi::SpiClient<'_>, cell: &Claimed) -> Result<Option<Outcome>, String> {
    let spec = &cell.spec;
    let row = client
        .update(&spec.lock_row_sql(), None, &[cell.row_key.clone().into()])
        .map_err(spi_err)?;
    if row.is_empty() {
        client
            .update(
                "DELETE FROM eventual.cells WHERE column_id = $1 AND row_key = $2 AND generation = $3",
                None,
                &[spec.column_id.into(), cell.row_key.clone().into(), cell.generation.into()],
            )
            .map_err(spi_err)?;
        return Ok(Some(Outcome::Gone));
    }
    let current = client
        .update(
            "SELECT generation, state FROM eventual.cells WHERE column_id = $1 AND row_key = $2 FOR UPDATE",
            None,
            &[spec.column_id.into(), cell.row_key.clone().into()],
        )
        .map_err(spi_err)?;
    if current.is_empty() {
        return Ok(Some(Outcome::Gone));
    }
    let first = current.first();
    let generation = first.get::<i64>(1).map_err(spi_err)?.unwrap_or(-1);
    let state = first.get::<String>(2).map_err(spi_err)?.unwrap_or_default();
    if generation != cell.generation || state != "running" {
        return Ok(Some(Outcome::Stale));
    }
    Ok(None)
}

fn mark_done(client: &mut pgrx::spi::SpiClient<'_>, cell: &Claimed, meta: Value, started: Instant) -> Result<(), String> {
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    client
        .update(
            "UPDATE eventual.cells
             SET state = 'done', finished_at = clock_timestamp(), duration_ms = $4,
                 worker_pid = NULL, last_error = NULL, result_meta = $5
             WHERE column_id = $1 AND row_key = $2 AND generation = $3",
            None,
            &[
                cell.spec.column_id.into(),
                cell.row_key.clone().into(),
                cell.generation.into(),
                elapsed_ms.into(),
                JsonB(meta).into(),
            ],
        )
        .map_err(spi_err)?;
    Ok(())
}

struct WritebackGuard;

impl WritebackGuard {
    fn new() -> Self {
        WRITEBACK.store(true, Ordering::SeqCst);
        WritebackGuard
    }
}

impl Drop for WritebackGuard {
    fn drop(&mut self) {
        WRITEBACK.store(false, Ordering::SeqCst);
    }
}

fn base_meta(cell: &Claimed) -> Value {
    json!({
        "processor": cell.spec.processor,
        "processor_version": cell.spec.processor_version,
        "kind": cell.spec.kind,
        "params": cell.spec.effective_params,
        "attempt": cell.attempts,
    })
}

fn write_sql(cell: &Claimed, started: Instant) -> Result<Outcome, String> {
    let spec = &cell.spec;
    let sql = spec.sql_writeback()?;
    Spi::connect_mut(|client| {
        if let Some(outcome) = lock_for_writeback(client, cell)? {
            return Ok(outcome);
        }
        if spec.strict {
            let missing = client
                .update(&spec.inputs_null_sql(), Some(1), &[cell.row_key.clone().into()])
                .map_err(spi_err)?
                .first()
                .get::<bool>(1)
                .map_err(spi_err)?
                .unwrap_or(false);
            if missing {
                client
                    .update(
                        "UPDATE eventual.cells SET state = 'waiting', worker_pid = NULL
                         WHERE column_id = $1 AND row_key = $2 AND generation = $3",
                        None,
                        &[spec.column_id.into(), cell.row_key.clone().into(), cell.generation.into()],
                    )
                    .map_err(spi_err)?;
                return Ok(Outcome::Waiting);
            }
        }
        {
            let _guard = WritebackGuard::new();
            as_role(spec.owner(), true, || {
                client.update(
                    &sql,
                    None,
                    &[cell.row_key.clone().into(), JsonB(spec.effective_params.clone()).into()],
                )
            })
            .map_err(spi_err)?;
        }
        mark_done(client, cell, base_meta(cell), started)?;
        Ok(Outcome::Done)
    })
}

fn http_timeout(cfg_ms: Option<u64>) -> Duration {
    Duration::from_millis(cfg_ms.unwrap_or(guc::HTTP_TIMEOUT_MS.get() as u64))
}

fn run_http(cell: &Claimed) -> Result<(Value, Value), String> {
    let spec = &cell.spec;
    let cfg = crate::http::HttpConfig::from_json(&spec.config).map_err(|e| format!("invalid http config: {e}"))?;
    let inputs = cell.inputs.as_ref().ok_or("inputs were not read")?;
    let body = crate::http::build_body(&cfg, inputs, &spec.effective_params, &spec.secrets)?;
    let headers = cfg.resolved_headers(&spec.secrets)?;
    let response = {
        let _blocked = SignalsBlocked::new();
        crate::http::send_json(&cfg.method, &cfg.url, &headers, &body, http_timeout(cfg.timeout_ms))
    }?;
    let value = crate::http::extract(&cfg, &response, spec.result_is_array)?;
    let mut meta = base_meta(cell);
    meta["url"] = json!(cfg.url);
    Ok((value, meta))
}

fn run_http_batch(cells: &[Claimed]) -> Result<Vec<(Value, Value)>, String> {
    let spec = &cells[0].spec;
    let cfg = crate::http::HttpConfig::from_json(&spec.config).map_err(|e| format!("invalid http config: {e}"))?;
    let rows: Vec<Vec<Value>> = cells
        .iter()
        .map(|c| c.inputs.clone().ok_or_else(|| "inputs were not read".to_string()))
        .collect::<Result<_, _>>()?;
    let body = crate::http::build_batch_body(&cfg, &rows, &spec.effective_params, &spec.secrets)?;
    let headers = cfg.resolved_headers(&spec.secrets)?;
    let response = {
        let _blocked = SignalsBlocked::new();
        crate::http::send_json(&cfg.method, &cfg.url, &headers, &body, http_timeout(cfg.timeout_ms))
    }?;
    let values = crate::http::extract_batch(&cfg, &response, cells.len(), spec.result_is_array)?;
    Ok(cells
        .iter()
        .zip(values)
        .map(|(cell, value)| {
            let mut meta = base_meta(cell);
            meta["url"] = json!(cfg.url);
            meta["batch"] = json!(cells.len());
            (value, meta)
        })
        .collect())
}

fn run_remote(cell: &Claimed) -> Result<(Value, Value), String> {
    if cell.spec.kind == "http" {
        return run_http(cell);
    }
    run_triton(cell)
}

fn run_triton(cell: &Claimed) -> Result<(Value, Value), String> {
    let spec = &cell.spec;
    let cfg = triton::TritonConfig::from_json(&spec.config).map_err(|e| format!("invalid triton config: {e}"))?;
    let inputs = cell.inputs.as_ref().ok_or("inputs were not read")?;
    let request = triton::build_request(&cfg, inputs, &spec.effective_params)?;
    let base_url = cfg
        .url
        .clone()
        .or_else(|| guc::string_setting(&guc::TRITON_URL))
        .ok_or("no Triton URL configured (set eventual.triton_url or the processor's \"url\")")?;
    let timeout = Duration::from_millis(cfg.timeout_ms.unwrap_or(guc::HTTP_TIMEOUT_MS.get() as u64));
    let response = {
        let _blocked = SignalsBlocked::new();
        triton::infer(&base_url, &cfg, &request, timeout)
    }?;
    let result = triton::parse_response(&cfg, &response, spec.result_is_array)?;
    let mut meta = base_meta(cell);
    meta["model_name"] = json!(result.model_name);
    meta["model_version"] = json!(result.model_version);
    Ok((result.value, meta))
}

fn write_json(cell: &Claimed, value: &Value, meta: Value, started: Instant) -> Result<Outcome, String> {
    let spec = &cell.spec;
    let mut record = serde_json::Map::new();
    record.insert(spec.target_name.clone(), value.clone());
    Spi::connect_mut(|client| {
        if let Some(outcome) = lock_for_writeback(client, cell)? {
            return Ok(outcome);
        }
        {
            let _guard = WritebackGuard::new();
            as_role(spec.owner(), true, || {
                client.update(
                    &spec.json_writeback(),
                    None,
                    &[cell.row_key.clone().into(), JsonB(Value::Object(record)).into()],
                )
            })
            .map_err(spi_err)?;
        }
        mark_done(client, cell, meta, started)?;
        Ok(Outcome::Done)
    })
}

fn record_failure(cell: &Claimed, error: &str, started: Instant) -> Result<(), String> {
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let message: String = error.chars().take(2000).collect();
    Spi::run_with_args(
        "UPDATE eventual.cells
         SET state = CASE WHEN attempts >= $4 THEN 'failed' ELSE 'pending' END,
             next_attempt_at = now() + make_interval(
                 secs => least(600.0, $5::float8 * power(2, greatest(attempts - 1, 0)) / 1000.0)),
             last_error = $6, finished_at = clock_timestamp(), duration_ms = $7, worker_pid = NULL
         WHERE column_id = $1 AND row_key = $2 AND generation = $3 AND state = 'running'",
        &[
            cell.spec.column_id.into(),
            cell.row_key.clone().into(),
            cell.generation.into(),
            cell.spec.max_attempts.into(),
            (guc::RETRY_BASE_MS.get() as f64).into(),
            message.into(),
            elapsed_ms.into(),
        ],
    )
    .map_err(spi_err)
}
