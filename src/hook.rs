use crate::declaration::{self, Declaration, ModArg};
use pgrx::pg_sys::{self, NodeTag};
use pgrx::prelude::*;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};

static mut PREV_HOOK: pg_sys::ProcessUtility_hook_type = None;
static INSTALLED: AtomicBool = AtomicBool::new(false);
static AT_PRELOAD: AtomicBool = AtomicBool::new(false);

pub fn install() {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    unsafe {
        PREV_HOOK = pg_sys::ProcessUtility_hook;
        pg_sys::ProcessUtility_hook = Some(eventual_process_utility);
        AT_PRELOAD.store(pg_sys::process_shared_preload_libraries_in_progress, Ordering::SeqCst);
    }
}

pub fn installed_at_preload() -> bool {
    AT_PRELOAD.load(Ordering::SeqCst)
}

struct PendingColumn {
    name: String,
    decl: Declaration,
}

#[allow(clippy::too_many_arguments)]
#[pg_guard]
unsafe extern "C-unwind" fn eventual_process_utility(
    pstmt: *mut pg_sys::PlannedStmt,
    query_string: *const c_char,
    read_only_tree: bool,
    context: pg_sys::ProcessUtilityContext::Type,
    params: pg_sys::ParamListInfo,
    query_env: *mut pg_sys::QueryEnvironment,
    dest: *mut pg_sys::DestReceiver,
    qc: *mut pg_sys::QueryCompletion,
) {
    let stmt = (*pstmt).utilityStmt;
    let tag = if stmt.is_null() { NodeTag::T_Invalid } else { (*stmt).type_ };
    let mentions_derived = match tag {
        NodeTag::T_CreateStmt => create_mentions_derived(stmt as *mut pg_sys::CreateStmt),
        NodeTag::T_AlterTableStmt => {
            let alter = stmt as *mut pg_sys::AlterTableStmt;
            alter_mentions_derived(alter) || !type_changed_columns(alter).is_empty()
        }
        NodeTag::T_RenameStmt => {
            let rename = stmt as *mut pg_sys::RenameStmt;
            (*rename).renameType == pg_sys::ObjectType::OBJECT_COLUMN && !(*rename).relation.is_null()
        }
        NodeTag::T_CreateTrigStmt => our_trigger_function(stmt as *mut pg_sys::CreateTrigStmt).is_some(),
        _ => false,
    };
    if !mentions_derived || !derived_type_available() {
        call_next(pstmt, query_string, read_only_tree, context, params, query_env, dest, qc);
        return;
    }

    let pstmt = if read_only_tree {
        pg_sys::copyObjectImpl(pstmt as *const c_void) as *mut pg_sys::PlannedStmt
    } else {
        pstmt
    };
    let stmt = (*pstmt).utilityStmt;
    let run = || unsafe { call_next(pstmt, query_string, false, context, params, query_env, dest, qc) };
    match tag {
        NodeTag::T_CreateStmt => handle_create(stmt as *mut pg_sys::CreateStmt, run),
        NodeTag::T_AlterTableStmt => handle_alter(stmt as *mut pg_sys::AlterTableStmt, run),
        NodeTag::T_RenameStmt => handle_rename(stmt as *mut pg_sys::RenameStmt, run),
        NodeTag::T_CreateTrigStmt => handle_create_trigger(stmt as *mut pg_sys::CreateTrigStmt, run),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_next(
    pstmt: *mut pg_sys::PlannedStmt,
    query_string: *const c_char,
    read_only_tree: bool,
    context: pg_sys::ProcessUtilityContext::Type,
    params: pg_sys::ParamListInfo,
    query_env: *mut pg_sys::QueryEnvironment,
    dest: *mut pg_sys::DestReceiver,
    qc: *mut pg_sys::QueryCompletion,
) {
    match PREV_HOOK {
        Some(prev) => pg_sys::ffi::pg_guard_ffi_boundary(|| {
            prev(pstmt, query_string, read_only_tree, context, params, query_env, dest, qc)
        }),
        None => pg_sys::standard_ProcessUtility(
            pstmt,
            query_string,
            read_only_tree,
            context,
            params,
            query_env,
            dest,
            qc,
        ),
    }
}

pub(crate) unsafe fn list_ptrs(list: *mut pg_sys::List) -> Vec<*mut c_void> {
    if list.is_null() {
        return Vec::new();
    }
    let len = (*list).length as usize;
    let elements = (*list).elements;
    (0..len).map(|i| (*elements.add(i)).ptr_value).collect()
}

unsafe fn node_tag(node: *mut c_void) -> NodeTag {
    if node.is_null() {
        NodeTag::T_Invalid
    } else {
        (*(node as *mut pg_sys::Node)).type_
    }
}

unsafe fn str_value(node: *mut c_void) -> Option<String> {
    if node_tag(node) != NodeTag::T_String {
        return None;
    }
    let s = node as *mut pg_sys::String;
    Some(CStr::from_ptr((*s).sval).to_string_lossy().into_owned())
}

unsafe fn type_name_parts(type_name: *mut pg_sys::TypeName) -> Vec<String> {
    if type_name.is_null() {
        return Vec::new();
    }
    list_ptrs((*type_name).names).into_iter().filter_map(|n| str_value(n)).collect()
}

unsafe fn is_derived_type_name(type_name: *mut pg_sys::TypeName) -> bool {
    let parts = type_name_parts(type_name);
    match parts.as_slice() {
        [name] => name == "derived",
        [schema, name] => schema == "eventual" && name == "derived",
        _ => false,
    }
}

unsafe fn column_def_is_derived(node: *mut c_void) -> bool {
    node_tag(node) == NodeTag::T_ColumnDef && is_derived_type_name((*(node as *mut pg_sys::ColumnDef)).typeName)
}

unsafe fn create_mentions_derived(stmt: *mut pg_sys::CreateStmt) -> bool {
    list_ptrs((*stmt).tableElts).into_iter().any(|n| column_def_is_derived(n))
}

unsafe fn alter_mentions_derived(stmt: *mut pg_sys::AlterTableStmt) -> bool {
    list_ptrs((*stmt).cmds).into_iter().any(|cmd| {
        let cmd = cmd as *mut pg_sys::AlterTableCmd;
        node_tag(cmd as *mut c_void) == NodeTag::T_AlterTableCmd && column_def_is_derived((*cmd).def as *mut c_void)
    })
}

fn derived_type_available() -> bool {
    unsafe {
        if !pg_sys::IsTransactionState() {
            return false;
        }
    }
    Spi::get_one::<bool>("SELECT to_regtype('eventual.derived') IS NOT NULL")
        .ok()
        .flatten()
        .unwrap_or(false)
}

fn unqualified_derived_is_ours() -> bool {
    Spi::get_one::<bool>("SELECT to_regtype('derived') IS NOT DISTINCT FROM to_regtype('eventual.derived')")
        .ok()
        .flatten()
        .unwrap_or(false)
}

unsafe fn mod_arg(node: *mut c_void) -> Result<ModArg, String> {
    match node_tag(node) {
        NodeTag::T_A_Const => {
            let c = node as *mut pg_sys::A_Const;
            if (*c).isnull {
                return Err("NULL is not allowed in a derived column declaration".into());
            }
            let val = &(*c).val;
            match val.node.type_ {
                NodeTag::T_String => Ok(ModArg::Str(CStr::from_ptr(val.sval.sval).to_string_lossy().into_owned())),
                NodeTag::T_Integer => Ok(ModArg::Int(val.ival.ival as i64)),
                NodeTag::T_Float => Ok(ModArg::Float(CStr::from_ptr(val.fval.fval).to_string_lossy().into_owned())),
                NodeTag::T_Boolean => Ok(ModArg::Bool(val.boolval.boolval)),
                other => Err(format!("unsupported constant in declaration ({other:?})")),
            }
        }
        NodeTag::T_ColumnRef => {
            let fields = list_ptrs((*(node as *mut pg_sys::ColumnRef)).fields);
            let names: Vec<String> = fields.into_iter().filter_map(|f| str_value(f)).collect();
            match names.as_slice() {
                [name] => Ok(ModArg::Ident(name.clone())),
                _ => Err("qualified names are not allowed in a derived column declaration".into()),
            }
        }
        NodeTag::T_TypeCast => mod_arg((*(node as *mut pg_sys::TypeCast)).arg as *mut c_void),
        other => Err(format!(
            "derived column declarations accept only identifiers and constants (found {other:?})"
        )),
    }
}

unsafe fn check_column_options(colname: &str, col: *mut pg_sys::ColumnDef) {
    let reject = |what: &str| -> ! {
        error!("derived column \"{colname}\" cannot have {what}: its value is NULL until computed")
    };
    if (*col).is_not_null {
        reject("a NOT NULL constraint");
    }
    if !(*col).raw_default.is_null() || !(*col).cooked_default.is_null() {
        reject("a DEFAULT");
    }
    if (*col).identity != 0 || (*col).generated != 0 {
        reject("IDENTITY or GENERATED");
    }
    if !(*(*col).typeName).arrayBounds.is_null() {
        error!("derived column \"{colname}\": eventual.derived(...) cannot be used as an array element type");
    }
    for constraint in list_ptrs((*col).constraints) {
        if node_tag(constraint) != NodeTag::T_Constraint {
            continue;
        }
        use pg_sys::ConstrType::*;
        match (*(constraint as *mut pg_sys::Constraint)).contype {
            CONSTR_NOTNULL => reject("a NOT NULL constraint"),
            CONSTR_DEFAULT => reject("a DEFAULT"),
            CONSTR_IDENTITY | CONSTR_GENERATED => reject("IDENTITY or GENERATED"),
            CONSTR_PRIMARY => reject("a PRIMARY KEY constraint"),
            _ => {}
        }
    }
}

fn processor_result_type(processor: &str) -> pg_sys::Oid {
    let found = Spi::get_one_with_args::<pg_sys::Oid>(
        "SELECT (SELECT result_type::oid FROM eventual.processors WHERE name = $1)",
        &[processor.into()],
    );
    match found {
        Ok(Some(oid)) => oid,
        Ok(None) => {
            let known = Spi::get_one::<String>(
                "SELECT coalesce(string_agg(name, ', ' ORDER BY name), 'none') FROM eventual.processors",
            )
            .ok()
            .flatten()
            .unwrap_or_default();
            ereport!(
                PgLogLevel::ERROR,
                PgSqlErrorCode::ERRCODE_UNDEFINED_OBJECT,
                format!("processor \"{processor}\" does not exist"),
                format!("Create it with eventual.create_processor(...). Existing processors: {known}.")
            );
            unreachable!()
        }
        Err(e) => error!("looking up processor \"{processor}\" failed: {e}"),
    }
}

unsafe fn rewrite_column(node: *mut c_void, unqualified_checked: &mut Option<bool>) -> Option<PendingColumn> {
    if !column_def_is_derived(node) {
        return None;
    }
    let col = node as *mut pg_sys::ColumnDef;
    let colname = CStr::from_ptr((*col).colname).to_string_lossy().into_owned();
    if type_name_parts((*col).typeName).len() == 1 {
        let ours = *unqualified_checked.get_or_insert_with(unqualified_derived_is_ours);
        if !ours {
            return None;
        }
    }
    check_column_options(&colname, col);
    let args = list_ptrs((*(*col).typeName).typmods)
        .into_iter()
        .map(|n| mod_arg(n))
        .collect::<Result<Vec<_>, _>>();
    let decl = match args.and_then(|a| declaration::parse(&a)) {
        Ok(decl) => decl,
        Err(message) => error!("invalid declaration for derived column \"{colname}\": {message}"),
    };
    let result_type = processor_result_type(&decl.processor);
    (*col).typeName = pg_sys::makeTypeNameFromOid(result_type, -1);
    Some(PendingColumn { name: colname, decl })
}

unsafe fn lookup_relation(relation: *mut pg_sys::RangeVar) -> pg_sys::Oid {
    pg_sys::RangeVarGetRelidExtended(
        relation,
        pg_sys::NoLock as pg_sys::LOCKMODE,
        pg_sys::RVROption::RVR_MISSING_OK,
        None,
        std::ptr::null_mut(),
    )
}

fn bind_columns(relid: pg_sys::Oid, columns: &[PendingColumn], backfill: bool) {
    unsafe { pg_sys::CommandCounterIncrement() };
    crate::security::as_extension_owner(|| {
        for column in columns {
            let params = pgrx::JsonB(serde_json::Value::Object(column.decl.params.clone()));
            let id = Spi::get_one_with_args::<i64>(
                "SELECT eventual._bind_column($1::oid::regclass, $2, $3, $4, $5)",
                &[
                    relid.into(),
                    column.name.clone().into(),
                    column.decl.processor.clone().into(),
                    column.decl.inputs.clone().into(),
                    params.into(),
                ],
            )
            .unwrap_or_else(|e| error!("binding derived column \"{}\" failed: {e}", column.name))
            .expect("_bind_column returns an id");
            if backfill {
                Spi::run_with_args("SELECT eventual._enqueue_all($1)", &[id.into()])
                    .unwrap_or_else(|e| error!("backfilling derived column \"{}\" failed: {e}", column.name));
            }
        }
        Spi::run_with_args("SELECT eventual._finalize_table($1::oid::regclass)", &[relid.into()])
            .unwrap_or_else(|e| error!("installing derived column triggers failed: {e}"));
    });
}

unsafe fn handle_create(stmt: *mut pg_sys::CreateStmt, run: impl FnOnce()) {
    let relation = (*stmt).relation;
    if (*relation).relpersistence == pg_sys::RELPERSISTENCE_TEMP as c_char {
        error!("derived columns are not supported on temporary tables (background workers cannot see them)");
    }
    if !(*stmt).partbound.is_null() || !(*stmt).inhRelations.is_null() {
        error!("derived columns are declared on the root of a partitioned table (not on partitions or inherited tables)");
    }
    if !(*stmt).ofTypename.is_null() {
        error!("derived columns are not supported on typed tables (CREATE TABLE ... OF type)");
    }
    let existed = lookup_relation(relation) != pg_sys::InvalidOid;
    let mut unqualified = None;
    let columns: Vec<PendingColumn> = list_ptrs((*stmt).tableElts)
        .into_iter()
        .filter_map(|n| rewrite_column(n, &mut unqualified))
        .collect();
    run();
    if existed || columns.is_empty() {
        return;
    }
    let relid = lookup_relation(relation);
    if relid == pg_sys::InvalidOid {
        error!("pg_eventual could not find the table it just created");
    }
    bind_columns(relid, &columns, false);
}

unsafe fn column_exists(relid: pg_sys::Oid, name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = $1 AND attname = $2 AND attnum > 0 AND NOT attisdropped)",
        &[relid.into(), name.into()],
    )
    .ok()
    .flatten()
    .unwrap_or(false)
}

fn table_has_bindings(relid: pg_sys::Oid) -> bool {
    crate::security::as_extension_owner(|| {
        Spi::get_one_with_args::<bool>(
            "SELECT EXISTS (SELECT 1 FROM eventual.derived_columns WHERE relid = $1::oid::regclass)",
            &[relid.into()],
        )
        .ok()
        .flatten()
        .unwrap_or(false)
    })
}

unsafe fn handle_rename(stmt: *mut pg_sys::RenameStmt, run: impl FnOnce()) {
    run();
    let relid = lookup_relation((*stmt).relation);
    if relid != pg_sys::InvalidOid && table_has_bindings(relid) {
        pg_sys::CommandCounterIncrement();
        crate::security::as_extension_owner(|| {
            Spi::run_with_args("SELECT eventual._sync_names($1::oid::regclass)", &[relid.into()])
                .unwrap_or_else(|e| error!("{e}"));
            Spi::run_with_args("SELECT eventual._finalize_table($1::oid::regclass)", &[relid.into()])
                .unwrap_or_else(|e| error!("{e}"));
        });
    }
}

unsafe fn our_trigger_function(stmt: *mut pg_sys::CreateTrigStmt) -> Option<String> {
    if (*stmt).replace || (*stmt).relation.is_null() {
        return None;
    }
    let parts: Vec<String> = list_ptrs((*stmt).funcname).into_iter().filter_map(|n| str_value(n)).collect();
    match parts.as_slice() {
        [schema, name] if schema == "eventual"
            && matches!(name.as_str(), "_row_trigger" | "_flush_trigger" | "_truncate_trigger") =>
        {
            Some(name.clone())
        }
        _ => None,
    }
}

unsafe fn handle_create_trigger(stmt: *mut pg_sys::CreateTrigStmt, run: impl FnOnce()) {
    let function = our_trigger_function(stmt).expect("checked by the caller");
    let relid = lookup_relation((*stmt).relation);
    if relid == pg_sys::InvalidOid || !table_has_bindings(relid) {
        run();
        return;
    }
    let spec = crate::security::as_extension_owner(|| {
        Spi::run_with_args("SELECT eventual._repair_table($1::oid::regclass)", &[relid.into()])
            .unwrap_or_else(|e| error!("{e}"));
        Spi::get_one_with_args::<String>("SELECT eventual._trigger_spec($1::oid::regclass)", &[relid.into()])
            .unwrap_or_else(|e| error!("{e}"))
            .unwrap_or_default()
    });
    if function == "_row_trigger" {
        let spec = std::ffi::CString::new(spec).expect("spec has no NUL bytes");
        let node = pg_sys::makeString(pg_sys::pstrdup(spec.as_ptr()));
        (*stmt).args = pg_sys::lappend(std::ptr::null_mut(), node as *mut c_void);
    }
    (*stmt).replace = true;
    run();
}

unsafe fn type_changed_columns(stmt: *mut pg_sys::AlterTableStmt) -> Vec<String> {
    list_ptrs((*stmt).cmds)
        .into_iter()
        .filter_map(|cmd| {
            let cmd = cmd as *mut pg_sys::AlterTableCmd;
            let changes_type = node_tag(cmd as *mut c_void) == NodeTag::T_AlterTableCmd
                && (*cmd).subtype == pg_sys::AlterTableType::AT_AlterColumnType
                && !column_def_is_derived((*cmd).def as *mut c_void)
                && !(*cmd).name.is_null();
            changes_type.then(|| CStr::from_ptr((*cmd).name).to_string_lossy().into_owned())
        })
        .collect()
}

unsafe fn handle_alter(stmt: *mut pg_sys::AlterTableStmt, run: impl FnOnce()) {
    let relid_before = lookup_relation((*stmt).relation);
    let type_changes = type_changed_columns(stmt);
    let mut unqualified = None;
    let mut columns = Vec::new();
    for cmd in list_ptrs((*stmt).cmds) {
        let cmd = cmd as *mut pg_sys::AlterTableCmd;
        if !column_def_is_derived((*cmd).def as *mut c_void) {
            continue;
        }
        if (*cmd).subtype != pg_sys::AlterTableType::AT_AddColumn {
            error!("eventual.derived(...) can only be used with ALTER TABLE ... ADD COLUMN");
        }
        let skip_existing = (*cmd).missing_ok && relid_before != pg_sys::InvalidOid && {
            let col = (*cmd).def as *mut pg_sys::ColumnDef;
            column_exists(relid_before, &CStr::from_ptr((*col).colname).to_string_lossy())
        };
        if let Some(pending) = rewrite_column((*cmd).def as *mut c_void, &mut unqualified) {
            if !skip_existing {
                columns.push(pending);
            }
        }
    }
    run();
    if columns.is_empty() && type_changes.is_empty() {
        return;
    }
    let relid = lookup_relation((*stmt).relation);
    if relid == pg_sys::InvalidOid {
        return;
    }
    if !type_changes.is_empty() {
        pg_sys::CommandCounterIncrement();
        crate::security::as_extension_owner(|| {
            Spi::run_with_args(
                "SELECT eventual._after_column_type_change($1::oid::regclass, $2::name[])",
                &[relid.into(), type_changes.clone().into()],
            )
            .unwrap_or_else(|e| error!("{e}"))
        });
    }
    if !columns.is_empty() {
        bind_columns(relid, &columns, true);
    }
}
