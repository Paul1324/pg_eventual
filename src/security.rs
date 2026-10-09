use pgrx::pg_sys::{self, PgTryBuilder};
use pgrx::prelude::*;
use std::panic::AssertUnwindSafe;

pub fn as_role<R>(role: pg_sys::Oid, restricted: bool, body: impl FnOnce() -> R) -> R {
    let mut saved_user = pg_sys::InvalidOid;
    let mut saved_context: i32 = 0;
    unsafe {
        pg_sys::GetUserIdAndSecContext(&mut saved_user, &mut saved_context);
        let mut context = saved_context | pg_sys::SECURITY_LOCAL_USERID_CHANGE as i32;
        if restricted {
            context |= pg_sys::SECURITY_RESTRICTED_OPERATION as i32;
        }
        pg_sys::SetUserIdAndSecContext(role, context);
    }
    PgTryBuilder::new(AssertUnwindSafe(body))
        .finally(|| unsafe { pg_sys::SetUserIdAndSecContext(saved_user, saved_context) })
        .execute()
}

pub fn extension_owner() -> pg_sys::Oid {
    Spi::get_one::<pg_sys::Oid>("SELECT (SELECT extowner FROM pg_catalog.pg_extension WHERE extname = 'pg_eventual')")
        .ok()
        .flatten()
        .unwrap_or_else(|| error!("pg_eventual is not installed in this database"))
}

pub fn as_extension_owner<R>(body: impl FnOnce() -> R) -> R {
    as_role(extension_owner(), false, body)
}

pub fn require_table_owner(relid: pg_sys::Oid) {
    let owns = unsafe { pg_sys::object_ownercheck(pg_sys::RelationRelationId, relid, pg_sys::GetUserId()) };
    if !owns {
        ereport!(
            PgLogLevel::ERROR,
            PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            "must be the owner of the table to manage its derived columns"
        );
    }
}

#[pg_extern(name = "_table_admin")]
fn table_admin(operation: &str, relid: pg_sys::Oid, column_name: Option<&str>, flag: bool) -> i64 {
    require_table_owner(relid);
    let sql = match operation {
        "refresh" => "SELECT eventual._refresh($1::oid::regclass, $2::name)",
        "retry_failed" => "SELECT eventual._retry_failed($1::oid::regclass, $2::name)",
        "drop_derived" => "SELECT eventual._drop_derived($1::oid::regclass, $2::name, $3)",
        other => error!("unknown pg_eventual operation \"{other}\""),
    };
    let column = column_name.map(str::to_owned);
    as_extension_owner(|| {
        Spi::get_one_with_args::<i64>(sql, &[relid.into(), column.into(), flag.into()])
            .unwrap_or_else(|e| error!("{e}"))
            .unwrap_or(0)
    })
}
