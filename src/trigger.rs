use crate::guc;
use crate::queue;
use crate::v1_function_info;
use pgrx::pg_sys::{self, NodeTag};
use pgrx::prelude::*;
use pgrx::PgTupleDesc;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CStr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

pub static WRITEBACK: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Deserialize)]
struct Spec {
    pk: Vec<i16>,
    #[serde(default)]
    pk_names: Vec<String>,
    cols: Vec<ColSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct ColSpec {
    id: i64,
    att: i16,
    #[serde(rename = "in")]
    inputs: Vec<i16>,
    strict: bool,
    #[serde(default)]
    name: String,
    #[serde(default)]
    in_names: Vec<String>,
}

thread_local! {
    static SPEC_CACHE: RefCell<HashMap<(u32, u32), (String, Rc<Spec>)>> = RefCell::new(HashMap::new());
}

fn remap(mut spec: Spec, attrs: &Attrs) -> Spec {
    let lookup = |name: &str, fallback: i16| -> i16 {
        let i = idx(fallback);
        if name.is_empty() || (i < attrs.names.len() && attrs.names[i] == name) {
            return fallback;
        }
        attrs.names.iter().position(|n| n == name).map_or(fallback, |i| (i + 1) as i16)
    };
    for (i, att) in spec.pk.iter_mut().enumerate() {
        *att = lookup(spec.pk_names.get(i).map(String::as_str).unwrap_or(""), *att);
    }
    for col in spec.cols.iter_mut() {
        col.att = lookup(&col.name, col.att);
        for (i, att) in col.inputs.iter_mut().enumerate() {
            *att = lookup(col.in_names.get(i).map(String::as_str).unwrap_or(""), *att);
        }
    }
    spec
}

fn spec_for(trigger_oid: pg_sys::Oid, relid: pg_sys::Oid, text: &str, attrs: &Attrs) -> Rc<Spec> {
    let key = (trigger_oid.to_u32(), relid.to_u32());
    SPEC_CACHE.with(|cache| {
        if let Some((cached_text, spec)) = cache.borrow().get(&key) {
            if cached_text == text {
                return spec.clone();
            }
        }
        let spec: Spec = serde_json::from_str(text)
            .unwrap_or_else(|e| error!("pg_eventual trigger has an invalid binding spec: {e}"));
        let spec = Rc::new(remap(spec, attrs));
        cache.borrow_mut().insert(key, (text.to_owned(), spec.clone()));
        spec
    })
}

struct Row {
    values: Vec<pg_sys::Datum>,
    nulls: Vec<bool>,
}

unsafe fn deform(tuple: pg_sys::HeapTuple, tupdesc: pg_sys::TupleDesc) -> Row {
    let natts = (*tupdesc).natts as usize;
    let mut values = vec![pg_sys::Datum::from(0); natts];
    let mut nulls = vec![true; natts];
    pg_sys::heap_deform_tuple(tuple, tupdesc, values.as_mut_ptr(), nulls.as_mut_ptr());
    Row { values, nulls }
}

struct Attrs {
    len: Vec<i16>,
    byval: Vec<bool>,
    typid: Vec<pg_sys::Oid>,
    names: Vec<String>,
}

unsafe fn attrs(tupdesc: pg_sys::TupleDesc) -> Attrs {
    let desc = PgTupleDesc::from_pg_unchecked(tupdesc);
    let mut out = Attrs { len: vec![], byval: vec![], typid: vec![], names: vec![] };
    for i in 0..desc.len() {
        let att = desc.get(i).expect("attribute in range");
        out.len.push(att.attlen);
        out.byval.push(att.attbyval);
        out.typid.push(att.atttypid);
        out.names.push(CStr::from_ptr(att.attname.data.as_ptr()).to_string_lossy().into_owned());
    }
    out
}

fn idx(attnum: i16) -> usize {
    (attnum - 1) as usize
}

unsafe fn same(attrs: &Attrs, a: &Row, b: &Row, i: usize) -> bool {
    match (a.nulls[i], b.nulls[i]) {
        (true, true) => true,
        (false, false) => pg_sys::datum_image_eq(a.values[i], b.values[i], attrs.byval[i], attrs.len[i] as i32),
        _ => false,
    }
}

unsafe fn row_key(attrs: &Attrs, row: &Row, pk: &[i16]) -> Vec<String> {
    pk.iter()
        .map(|&att| {
            let i = idx(att);
            if row.nulls[i] {
                error!("primary key column \"{}\" is NULL", attrs.names[i]);
            }
            let mut output_fn = pg_sys::InvalidOid;
            let mut varlena = false;
            pg_sys::getTypeOutputInfo(attrs.typid[i], &mut output_fn, &mut varlena);
            let text = pg_sys::OidOutputFunctionCall(output_fn, row.values[i]);
            let owned = CStr::from_ptr(text).to_string_lossy().into_owned();
            pg_sys::pfree(text.cast());
            owned
        })
        .collect()
}

fn state_for(col: &ColSpec, row: &Row) -> &'static str {
    if col.strict && col.inputs.iter().any(|&a| row.nulls[idx(a)]) {
        "waiting"
    } else {
        "pending"
    }
}

fn all_ids(spec: &Spec) -> Rc<Vec<i64>> {
    Rc::new(spec.cols.iter().map(|c| c.id).collect())
}

v1_function_info!(pg_finfo_eventual_row_trigger);

#[unsafe(no_mangle)]
#[pg_guard]
pub unsafe extern "C-unwind" fn eventual_row_trigger(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let context = (*fcinfo).context;
    if context.is_null() || (*context).type_ != NodeTag::T_TriggerData {
        error!("eventual._row_trigger() can only be called as a trigger");
    }
    let tdata = context as *mut pg_sys::TriggerData;
    let trigger = (*tdata).tg_trigger;
    if (*trigger).tgnargs != 1 {
        error!("eventual._row_trigger() expects one argument (the binding spec)");
    }
    let spec_text = CStr::from_ptr(*(*trigger).tgargs).to_string_lossy();
    let event = (*tdata).tg_event;
    if event & pg_sys::TRIGGER_EVENT_ROW == 0 {
        error!("eventual._row_trigger() must be a FOR EACH ROW trigger");
    }
    let relation = (*tdata).tg_relation;
    let tupdesc = (*relation).rd_att;
    let attrs = attrs(tupdesc);
    let spec = spec_for((*trigger).tgoid, (*relation).rd_id, &spec_text, &attrs);
    let op = event & pg_sys::TRIGGER_EVENT_OPMASK;
    let before = event & pg_sys::TRIGGER_EVENT_TIMINGMASK == pg_sys::TRIGGER_EVENT_BEFORE;

    if before {
        let result = match op {
            pg_sys::TRIGGER_EVENT_INSERT => before_insert(&spec, tdata, tupdesc),
            pg_sys::TRIGGER_EVENT_UPDATE => before_update(&spec, &attrs, tdata, tupdesc),
            _ => (*tdata).tg_trigtuple,
        };
        return pg_sys::Datum::from(result);
    }
    match op {
        pg_sys::TRIGGER_EVENT_INSERT => {
            let row = deform((*tdata).tg_trigtuple, tupdesc);
            let trust = guc::TRUST_INSERTED_VALUES.get();
            let key = Rc::new(row_key(&attrs, &row, &spec.pk));
            for col in &spec.cols {
                let state = if trust && !row.nulls[idx(col.att)] { "done" } else { state_for(col, &row) };
                queue::upsert(col.id, key.clone(), state);
            }
        }
        pg_sys::TRIGGER_EVENT_UPDATE => {
            let old = deform((*tdata).tg_trigtuple, tupdesc);
            let new = deform((*tdata).tg_newtuple, tupdesc);
            let old_key = Rc::new(row_key(&attrs, &old, &spec.pk));
            let new_key = Rc::new(row_key(&attrs, &new, &spec.pk));
            if old_key != new_key {
                queue::move_key(all_ids(&spec), old_key, new_key.clone());
            }
            for col in &spec.cols {
                if col.inputs.iter().any(|&a| !same(&attrs, &old, &new, idx(a))) {
                    queue::upsert(col.id, new_key.clone(), state_for(col, &new));
                }
            }
        }
        pg_sys::TRIGGER_EVENT_DELETE => {
            let old = deform((*tdata).tg_trigtuple, tupdesc);
            queue::delete(all_ids(&spec), Rc::new(row_key(&attrs, &old, &spec.pk)));
        }
        _ => {}
    }
    pg_sys::Datum::from(0)
}

unsafe fn modify(
    tuple: pg_sys::HeapTuple,
    tupdesc: pg_sys::TupleDesc,
    row: &mut Row,
    replace: &mut [bool],
) -> pg_sys::HeapTuple {
    if !replace.iter().any(|r| *r) {
        return tuple;
    }
    pg_sys::heap_modify_tuple(tuple, tupdesc, row.values.as_mut_ptr(), row.nulls.as_mut_ptr(), replace.as_mut_ptr())
}

unsafe fn before_insert(spec: &Spec, tdata: *mut pg_sys::TriggerData, tupdesc: pg_sys::TupleDesc) -> pg_sys::HeapTuple {
    let tuple = (*tdata).tg_trigtuple;
    let mut row = deform(tuple, tupdesc);
    let mut replace = vec![false; row.values.len()];
    if guc::TRUST_INSERTED_VALUES.get() {
        return tuple;
    }
    for col in &spec.cols {
        let i = idx(col.att);
        if !row.nulls[i] {
            row.nulls[i] = true;
            replace[i] = true;
        }
    }
    modify(tuple, tupdesc, &mut row, &mut replace)
}

unsafe fn before_update(
    spec: &Spec,
    attrs: &Attrs,
    tdata: *mut pg_sys::TriggerData,
    tupdesc: pg_sys::TupleDesc,
) -> pg_sys::HeapTuple {
    let old = deform((*tdata).tg_trigtuple, tupdesc);
    let tuple = (*tdata).tg_newtuple;
    let mut new = deform(tuple, tupdesc);
    let mut replace = vec![false; new.values.len()];
    let writeback = WRITEBACK.load(Ordering::SeqCst);
    for col in &spec.cols {
        let i = idx(col.att);
        let inputs_changed = col.inputs.iter().any(|&a| !same(attrs, &old, &new, idx(a)));
        if inputs_changed {
            if !new.nulls[i] {
                new.nulls[i] = true;
                replace[i] = true;
            }
        } else if !writeback && !same(attrs, &old, &new, i) {
            ereport!(
                PgLogLevel::ERROR,
                PgSqlErrorCode::ERRCODE_GENERATED_ALWAYS,
                format!("column \"{}\" is derived by pg_eventual and cannot be updated directly", attrs.names[i]),
                "Update its input columns instead, or use eventual.refresh() to recompute it."
            );
        }
    }
    modify(tuple, tupdesc, &mut new, &mut replace)
}
