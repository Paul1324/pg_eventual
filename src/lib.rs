use pgrx::bgworkers::{BackgroundWorkerBuilder, BgWorkerStartTime};
use pgrx::prelude::*;
use std::time::Duration;

pub mod declaration;
pub mod guc;
pub mod http;
mod hook;
pub mod params;
mod processors;
mod queue;
mod security;
pub mod triton;
mod trigger;
mod typmod;
mod worker;

::pgrx::pg_module_magic!(name, version);

pgrx::extension_sql_file!("../sql/catalog.sql", name = "catalog", bootstrap);
pgrx::extension_sql_file!("../sql/api.sql", name = "api", finalize);

#[allow(non_snake_case)]
#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    guc::init();
    hook::install();

    if unsafe { pg_sys::process_shared_preload_libraries_in_progress } {
        queue::init_shmem();
        let per_database = guc::WORKERS.get();
        let mut slot = 0i32;
        for database in guc::databases() {
            for index in 0..per_database {
                if slot as usize >= queue::MAX_WORKERS {
                    warning!("pg_eventual: at most {} workers in total; not starting more", queue::MAX_WORKERS);
                    break;
                }
                BackgroundWorkerBuilder::new(&format!("pg_eventual worker {index} ({database})"))
                    .set_type("pg_eventual worker")
                    .set_library("pg_eventual")
                    .set_function("eventual_worker_main")
                    .set_argument(slot.into_datum())
                    .set_extra(&database)
                    .enable_spi_access()
                    .set_start_time(BgWorkerStartTime::RecoveryFinished)
                    .set_restart_time(Some(Duration::from_secs(5)))
                    .load();
                slot += 1;
            }
        }
    }
}

#[pg_extern(name = "_preloaded")]
fn preloaded() -> bool {
    hook::installed_at_preload()
}

// Latch timeouts use the monotonic clock; pg_sleep() is stretched by wall-clock jumps.
#[pg_extern(name = "_nap", volatile, parallel_safe)]
fn nap(ms: i32) {
    unsafe {
        pg_sys::WaitLatch(
            pg_sys::MyLatch,
            (pg_sys::WL_LATCH_SET | pg_sys::WL_TIMEOUT | pg_sys::WL_EXIT_ON_PM_DEATH) as i32,
            ms.max(0) as std::ffi::c_long,
            pg_sys::PG_WAIT_EXTENSION,
        );
        pg_sys::ResetLatch(pg_sys::MyLatch);
    }
    pgrx::check_for_interrupts!();
}

#[pg_extern(name = "_monotonic_seconds", volatile, parallel_safe)]
fn monotonic_seconds() -> f64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

#[pg_extern(name = "_merge_params", immutable, parallel_safe)]
fn merge_params_sql(defaults: pgrx::JsonB, overrides: pgrx::JsonB) -> pgrx::JsonB {
    match params::merge(&defaults.0, &overrides.0) {
        Ok(value) => pgrx::JsonB(value),
        Err(message) => error!("{message}"),
    }
}

#[pg_extern(name = "_validate_triton_config", immutable, parallel_safe)]
fn validate_triton_config(config: pgrx::JsonB, inputs: i32, defaults: pgrx::JsonB) -> bool {
    match triton::TritonConfig::from_json(&config.0) {
        Ok(cfg) => {
            if let Err(message) = cfg.check_parameters(&defaults.0) {
                error!("invalid triton config: {message}");
            }
            if cfg.inputs.len() != inputs as usize {
                error!(
                    "triton config lists {} input tensors but the processor declares {} input types",
                    cfg.inputs.len(),
                    inputs
                );
            }
            true
        }
        Err(message) => error!("invalid triton config: {message}"),
    }
}

#[pg_extern(name = "_validate_http_config", immutable, parallel_safe)]
fn validate_http_config(config: pgrx::JsonB, inputs: i32, defaults: pgrx::JsonB) -> bool {
    match http::HttpConfig::from_json(&config.0).and_then(|cfg| cfg.check(inputs as usize, &defaults.0)) {
        Ok(()) => true,
        Err(message) => error!("invalid http config: {message}"),
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["shared_preload_libraries = 'pg_eventual'"]
    }
}
