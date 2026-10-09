use pgrx::pg_sys;
use pgrx::prelude::*;

#[macro_export]
macro_rules! v1_function_info {
    ($finfo:ident) => {
        #[unsafe(no_mangle)]
        #[doc(hidden)]
        pub extern "C" fn $finfo() -> &'static ::pgrx::pg_sys::Pg_finfo_record {
            const V1: ::pgrx::pg_sys::Pg_finfo_record = ::pgrx::pg_sys::Pg_finfo_record { api_version: 1 };
            &V1
        }
    };
}

v1_function_info!(pg_finfo_eventual_derived_typmod_in);

#[unsafe(no_mangle)]
#[pg_guard]
pub unsafe extern "C-unwind" fn eventual_derived_typmod_in(_fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let preloaded = crate::hook::installed_at_preload();
    if preloaded {
        error!(
            "eventual.derived(...) can only be used as a column type in CREATE TABLE or ALTER TABLE ... ADD COLUMN"
        );
    } else {
        ereport!(
            PgLogLevel::ERROR,
            PgSqlErrorCode::ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE,
            "pg_eventual is not loaded through shared_preload_libraries",
            "Add pg_eventual to shared_preload_libraries and restart PostgreSQL to declare derived columns."
        );
    }
    pg_sys::Datum::from(0)
}
