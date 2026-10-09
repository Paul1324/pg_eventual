use crate::v1_function_info;
use pgrx::pg_sys::{self, NodeTag};
use pgrx::prelude::*;
use pgrx::{pg_shmem_init, PGRXSharedMemory, PgAtomic};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

pub const MAX_WORKERS: usize = 64;

pub struct WorkerSlots {
    procs: [AtomicI32; MAX_WORKERS],
    databases: [AtomicU32; MAX_WORKERS],
}

impl Default for WorkerSlots {
    fn default() -> Self {
        WorkerSlots {
            procs: std::array::from_fn(|_| AtomicI32::new(-1)),
            databases: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }
}

unsafe impl PGRXSharedMemory for WorkerSlots {}

static SLOTS: PgAtomic<WorkerSlots> = unsafe { PgAtomic::new(c"pg_eventual_worker_slots") };
static SHMEM_READY: AtomicBool = AtomicBool::new(false);

pub fn init_shmem() {
    pg_shmem_init!(SLOTS = WorkerSlots::default());
    SHMEM_READY.store(true, Ordering::SeqCst);
}

pub fn register_worker(slot: i32, proc_number: i32, database: pg_sys::Oid) {
    if SHMEM_READY.load(Ordering::SeqCst) && (slot as usize) < MAX_WORKERS {
        let slots = SLOTS.get();
        slots.databases[slot as usize].store(database.to_u32(), Ordering::SeqCst);
        slots.procs[slot as usize].store(proc_number, Ordering::SeqCst);
    }
}

fn wake_workers() {
    if !SHMEM_READY.load(Ordering::SeqCst) {
        return;
    }
    unsafe {
        let global = pg_sys::ProcGlobal;
        if global.is_null() {
            return;
        }
        let database = pg_sys::MyDatabaseId.to_u32();
        let slots = SLOTS.get();
        for (slot, db) in slots.procs.iter().zip(slots.databases.iter()) {
            let number = slot.load(Ordering::SeqCst);
            if number >= 0 && db.load(Ordering::SeqCst) == database {
                let proc_ = (*global).allProcs.add(number as usize);
                pg_sys::SetLatch(&raw mut (*proc_).procLatch);
            }
        }
    }
}

enum Op {
    Upsert { column_id: i64, key: Rc<Vec<String>>, state: &'static str },
    Delete { ids: Rc<Vec<i64>>, key: Rc<Vec<String>> },
    Move { ids: Rc<Vec<i64>>, old: Rc<Vec<String>>, new: Rc<Vec<String>> },
}

struct Entry {
    subxact: pg_sys::SubTransactionId,
    op: Op,
}

thread_local! {
    static BUFFER: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static WAKE_ON_COMMIT: Cell<bool> = const { Cell::new(false) };
    static CALLBACKS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

fn ensure_callbacks() {
    CALLBACKS_REGISTERED.with(|done| {
        if !done.get() {
            unsafe {
                pg_sys::RegisterXactCallback(Some(xact_callback), std::ptr::null_mut());
                pg_sys::RegisterSubXactCallback(Some(subxact_callback), std::ptr::null_mut());
            }
            done.set(true);
        }
    });
}

fn push(op: Op) {
    ensure_callbacks();
    let subxact = unsafe { pg_sys::GetCurrentSubTransactionId() };
    BUFFER.with(|b| b.borrow_mut().push(Entry { subxact, op }));
}

pub fn upsert(column_id: i64, key: Rc<Vec<String>>, state: &'static str) {
    push(Op::Upsert { column_id, key, state });
}

pub fn delete(ids: Rc<Vec<i64>>, key: Rc<Vec<String>>) {
    push(Op::Delete { ids, key });
}

pub fn move_key(ids: Rc<Vec<i64>>, old: Rc<Vec<String>>, new: Rc<Vec<String>>) {
    push(Op::Move { ids, old, new });
}

#[pg_guard]
unsafe extern "C-unwind" fn xact_callback(event: pg_sys::XactEvent::Type, _arg: *mut c_void) {
    use pg_sys::XactEvent::*;
    match event {
        XACT_EVENT_PRE_COMMIT | XACT_EVENT_PARALLEL_PRE_COMMIT => {
            if BUFFER.with(|b| !b.borrow().is_empty()) {
                // No snapshot is active at pre-commit; SPI needs one.
                pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
                crate::security::as_extension_owner(flush);
                pg_sys::PopActiveSnapshot();
            }
        }
        XACT_EVENT_COMMIT | XACT_EVENT_PARALLEL_COMMIT => {
            if WAKE_ON_COMMIT.with(|w| w.replace(false)) {
                wake_workers();
            }
            BUFFER.with(|b| b.borrow_mut().clear());
        }
        XACT_EVENT_ABORT | XACT_EVENT_PARALLEL_ABORT | XACT_EVENT_PREPARE => {
            WAKE_ON_COMMIT.with(|w| w.set(false));
            BUFFER.with(|b| b.borrow_mut().clear());
        }
        _ => {}
    }
}

#[pg_guard]
unsafe extern "C-unwind" fn subxact_callback(
    event: pg_sys::SubXactEvent::Type,
    my_subid: pg_sys::SubTransactionId,
    _parent: pg_sys::SubTransactionId,
    _arg: *mut c_void,
) {
    if event == pg_sys::SubXactEvent::SUBXACT_EVENT_ABORT_SUB {
        BUFFER.with(|b| b.borrow_mut().retain(|e| e.subxact < my_subid));
    }
}

const UPSERT_SQL: &str = "
INSERT INTO eventual.cells AS c (column_id, row_key, state, finished_at, result_meta)
SELECT ($1::bigint[])[i],
       ($2::text[])[(i - 1) * $4 + 1 : i * $4],
       ($3::text[])[i],
       CASE WHEN ($3::text[])[i] = 'done' THEN now() END,
       CASE WHEN ($3::text[])[i] = 'done' THEN '{\"source\": \"inserted\"}'::jsonb END
FROM generate_series(1, cardinality($1::bigint[])) AS i
ON CONFLICT (column_id, row_key) DO UPDATE
SET state = EXCLUDED.state,
    generation = c.generation + 1,
    attempts = 0,
    enqueued_at = now(),
    next_attempt_at = now(),
    started_at = NULL,
    finished_at = EXCLUDED.finished_at,
    worker_pid = NULL,
    last_error = NULL,
    duration_ms = NULL,
    result_meta = EXCLUDED.result_meta";

const DELETE_SQL: &str = "
DELETE FROM eventual.cells c
USING (SELECT ($2::text[])[(i - 1) * $3 + 1 : i * $3] AS row_key
       FROM generate_series(1, cardinality($2::text[]) / $3) AS i) d
WHERE c.column_id = ANY ($1::bigint[]) AND c.row_key = d.row_key";

const MOVE_SQL: &str = "UPDATE eventual.cells SET row_key = $3 WHERE column_id = ANY ($1) AND row_key = $2";

pub fn flush() {
    let entries = BUFFER.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if entries.is_empty() {
        return;
    }
    let mut moves = Vec::new();
    let mut deletes: HashMap<(Rc<Vec<i64>>, usize), Vec<String>> = HashMap::new();
    let mut upserts: HashMap<(i64, Rc<Vec<String>>), &'static str> = HashMap::new();
    let mut upsert_order: Vec<(i64, Rc<Vec<String>>)> = Vec::new();
    for entry in entries {
        match entry.op {
            Op::Move { ids, old, new } => moves.push((ids, old, new)),
            Op::Delete { ids, key } => {
                deletes.entry((ids, key.len())).or_default().extend(key.iter().cloned());
            }
            Op::Upsert { column_id, key, state } => {
                let slot = (column_id, key);
                if upserts.insert(slot.clone(), state).is_none() {
                    upsert_order.push(slot);
                }
            }
        }
    }
    let mut wake = false;
    Spi::connect_mut(|client| -> Result<(), pgrx::spi::SpiError> {
        for (ids, old, new) in moves {
            client.update(MOVE_SQL, None, &[(*ids).clone().into(), (*old).clone().into(), (*new).clone().into()])?;
        }
        for ((ids, key_len), keys) in deletes {
            if key_len > 0 {
                client.update(DELETE_SQL, None, &[(*ids).clone().into(), keys.into(), (key_len as i32).into()])?;
            }
        }
        let mut groups: HashMap<usize, (Vec<i64>, Vec<String>, Vec<String>)> = HashMap::new();
        for slot in upsert_order {
            let state = upserts[&slot];
            wake |= state == "pending";
            let (column_id, key) = slot;
            let group = groups.entry(key.len()).or_default();
            group.0.push(column_id);
            group.1.extend(key.iter().cloned());
            group.2.push(state.to_string());
        }
        for (key_len, (ids, keys, states)) in groups {
            client.update(UPSERT_SQL, None, &[ids.into(), keys.into(), states.into(), (key_len as i32).into()])?;
        }
        Ok(())
    })
    .unwrap_or_else(|e| error!("pg_eventual could not record derived column work: {e}"));
    if wake {
        WAKE_ON_COMMIT.with(|w| w.set(true));
    }
}

#[pg_extern(name = "_wake_workers_on_commit")]
fn wake_workers_on_commit() {
    ensure_callbacks();
    WAKE_ON_COMMIT.with(|w| w.set(true));
}

v1_function_info!(pg_finfo_eventual_flush_trigger);

#[unsafe(no_mangle)]
#[pg_guard]
pub unsafe extern "C-unwind" fn eventual_flush_trigger(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let context = (*fcinfo).context;
    if context.is_null() || (*context).type_ != NodeTag::T_TriggerData {
        error!("eventual._flush_trigger() can only be called as a trigger");
    }
    flush();
    pg_sys::Datum::from(0)
}
