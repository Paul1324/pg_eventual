use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use pgrx::pg_sys;
use std::ffi::CString;

pub static DATABASE: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(Some(c"postgres"));
pub static ROLE: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);
pub static WORKERS: GucSetting<i32> = GucSetting::<i32>::new(1);
pub static POLL_INTERVAL_MS: GucSetting<i32> = GucSetting::<i32>::new(1_000);
pub static BATCH_SIZE: GucSetting<i32> = GucSetting::<i32>::new(32);
pub static TRITON_URL: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"http://localhost:8000"));
pub static HTTP_TIMEOUT_MS: GucSetting<i32> = GucSetting::<i32>::new(30_000);
pub static RETRY_BASE_MS: GucSetting<i32> = GucSetting::<i32>::new(1_000);
pub static TRUST_INSERTED_VALUES: GucSetting<bool> = GucSetting::<bool>::new(false);

pub fn init() {
    GucRegistry::define_string_guc(
        c"eventual.database",
        c"Database(s) served by the pg_eventual background workers (comma-separated list).",
        c"Each listed database gets eventual.workers workers; the extension must be created in it.",
        &DATABASE,
        GucContext::Postmaster,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"eventual.role",
        c"Role the pg_eventual background workers use (default: bootstrap superuser).",
        c"",
        &ROLE,
        GucContext::Postmaster,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"eventual.workers",
        c"Number of pg_eventual background workers.",
        c"",
        &WORKERS,
        0,
        64,
        GucContext::Postmaster,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"eventual.poll_interval_ms",
        c"Fallback polling interval of idle workers (they are also woken when work is committed).",
        c"",
        &POLL_INTERVAL_MS,
        10,
        60_000,
        GucContext::Sighup,
        GucFlags::UNIT_MS,
    );
    GucRegistry::define_int_guc(
        c"eventual.batch_size",
        c"Maximum number of cells a worker claims at once.",
        c"",
        &BATCH_SIZE,
        1,
        1000,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"eventual.triton_url",
        c"Default Triton HTTP endpoint for processors that do not set \"url\".",
        c"",
        &TRITON_URL,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"eventual.http_timeout_ms",
        c"Default total timeout of one remote processor request.",
        c"",
        &HTTP_TIMEOUT_MS,
        100,
        3_600_000,
        GucContext::Sighup,
        GucFlags::UNIT_MS,
    );
    GucRegistry::define_int_guc(
        c"eventual.retry_base_ms",
        c"Base delay of the exponential retry backoff after a failed attempt.",
        c"",
        &RETRY_BASE_MS,
        0,
        3_600_000,
        GucContext::Sighup,
        GucFlags::UNIT_MS,
    );
    GucRegistry::define_bool_guc(
        c"eventual.trust_inserted_values",
        c"Keep non-NULL derived values supplied by INSERT instead of recomputing them.",
        c"Useful when restoring or migrating data that already contains computed values.",
        &TRUST_INSERTED_VALUES,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe { pg_sys::MarkGUCPrefixReserved(c"eventual".as_ptr()) };
}

pub fn string_setting(setting: &GucSetting<Option<CString>>) -> Option<String> {
    setting.get().map(|value| value.to_string_lossy().into_owned())
}

pub fn databases() -> Vec<String> {
    string_setting(&DATABASE)
        .unwrap_or_default()
        .split(',')
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .collect()
}
