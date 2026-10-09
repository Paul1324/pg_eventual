use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct ColumnSpec {
    pub column_id: i64,
    pub table_sql: String,
    pub owner_oid: u32,
    pub target_sql: String,
    pub target_name: String,
    pub inputs_sql: Vec<String>,
    pub pk_sql: Vec<String>,
    pub pk_types: Vec<String>,
    pub processor: String,
    #[serde(default)]
    pub processor_version: i32,
    pub kind: String,
    pub fn_sql: Option<String>,
    pub input_types: Vec<Option<String>>,
    pub takes_params: bool,
    pub config: Value,
    pub effective_params: Value,
    pub strict: bool,
    pub max_attempts: i32,
    pub result_is_array: bool,
    #[serde(skip)]
    pub secrets: std::collections::BTreeMap<String, String>,
}

impl ColumnSpec {
    pub fn owner(&self) -> pgrx::pg_sys::Oid {
        pgrx::pg_sys::Oid::from(self.owner_oid)
    }

    pub fn where_clause(&self, param: usize) -> String {
        self.pk_sql
            .iter()
            .zip(&self.pk_types)
            .enumerate()
            .map(|(i, (column, ty))| format!("{column} = (${param}::text[])[{}]::{ty}", i + 1))
            .collect::<Vec<_>>()
            .join(" AND ")
    }

    pub fn lock_row_sql(&self) -> String {
        format!("SELECT 1 FROM {} WHERE {} FOR UPDATE", self.table_sql, self.where_clause(1))
    }

    pub fn read_inputs_sql(&self) -> String {
        format!(
            "SELECT jsonb_build_array({}) FROM {} WHERE {}",
            self.inputs_sql.join(", "),
            self.table_sql,
            self.where_clause(1)
        )
    }

    pub fn inputs_null_sql(&self) -> String {
        let checks: Vec<String> = self.inputs_sql.iter().map(|c| format!("{c} IS NULL")).collect();
        format!("SELECT {} FROM {} WHERE {}", checks.join(" OR "), self.table_sql, self.where_clause(1))
    }

    pub fn sql_call(&self) -> Result<String, String> {
        self.sql_call_with("", 2)
    }

    pub fn sql_call_with(&self, prefix: &str, params: usize) -> Result<String, String> {
        let function = self.fn_sql.as_ref().ok_or("sql processor without a function")?;
        let mut args: Vec<String> = self
            .inputs_sql
            .iter()
            .zip(&self.input_types)
            .map(|(column, ty)| match ty {
                Some(ty) => format!("{prefix}{column}::{ty}"),
                None => format!("{prefix}{column}"),
            })
            .collect();
        if self.takes_params {
            args.push(format!("${params}::jsonb"));
        }
        Ok(format!("{function}({})", args.join(", ")))
    }

    pub fn keys_source(&self, keys: usize, count: usize) -> String {
        let n = self.pk_sql.len();
        format!(
            "(SELECT ev_i, (${keys}::text[])[(ev_i - 1) * {n} + 1 : ev_i * {n}] AS ev_key \
             FROM generate_series(1, ${count}) AS ev_i)"
        )
    }

    fn key_join(&self, table: &str, keys: &str) -> String {
        self.pk_sql
            .iter()
            .zip(&self.pk_types)
            .enumerate()
            .map(|(j, (column, ty))| format!("{table}.{column} = {keys}.ev_key[{}]::{ty}", j + 1))
            .collect::<Vec<_>>()
            .join(" AND ")
    }

    pub fn lock_rows_batch_sql(&self) -> String {
        let order: Vec<String> = self.pk_sql.iter().map(|c| format!("t.{c}")).collect();
        format!(
            "SELECT k.ev_i FROM {keys} k JOIN {table} t ON {join} ORDER BY {order} FOR UPDATE OF t SKIP LOCKED",
            keys = self.keys_source(1, 2),
            table = self.table_sql,
            join = self.key_join("t", "k"),
            order = order.join(", ")
        )
    }

    pub fn rows_exist_batch_sql(&self) -> String {
        format!(
            "SELECT k.ev_i FROM {keys} k JOIN {table} t ON {join}",
            keys = self.keys_source(1, 2),
            table = self.table_sql,
            join = self.key_join("t", "k")
        )
    }

    pub fn sql_batch_writeback(&self) -> Result<String, String> {
        let strict = if self.strict {
            let checks: Vec<String> = self.inputs_sql.iter().map(|c| format!("t.{c} IS NULL")).collect();
            format!(" AND NOT ({})", checks.join(" OR "))
        } else {
            String::new()
        };
        Ok(format!(
            "UPDATE {table} t SET {target} = {call} FROM {keys} k WHERE {join}{strict} RETURNING k.ev_i",
            table = self.table_sql,
            target = self.target_sql,
            call = self.sql_call_with("t.", 3)?,
            keys = self.keys_source(1, 2),
            join = self.key_join("t", "k"),
        ))
    }

    pub fn sql_writeback(&self) -> Result<String, String> {
        Ok(format!(
            "UPDATE {} SET {} = {} WHERE {}",
            self.table_sql,
            self.target_sql,
            self.sql_call()?,
            self.where_clause(1)
        ))
    }

    pub fn json_batch_writeback(&self) -> String {
        let n = self.pk_sql.len();
        format!(
            "UPDATE {table} t SET {target} = (jsonb_populate_record(NULL::{table}, k.ev_val)).{target} \
             FROM (SELECT ev_i, ($1::text[])[(ev_i - 1) * {n} + 1 : ev_i * {n}] AS ev_key, \
                          $3::jsonb -> (ev_i - 1) AS ev_val \
                   FROM generate_series(1, $2) AS ev_i) k \
             WHERE {join} RETURNING k.ev_i",
            table = self.table_sql,
            target = self.target_sql,
            join = self.key_join("t", "k"),
        )
    }

    pub fn json_writeback(&self) -> String {
        format!(
            "UPDATE {table} SET {target} = (jsonb_populate_record(NULL::{table}, $2::jsonb)).{target} WHERE {cond}",
            table = self.table_sql,
            target = self.target_sql,
            cond = self.where_clause(1)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> ColumnSpec {
        serde_json::from_value(json!({
            "column_id": 7,
            "table_sql": "public.docs",
            "owner_oid": 10,
            "target_sql": "\"Summary\"",
            "target_name": "Summary",
            "inputs_sql": ["title", "body"],
            "pk_sql": ["tenant", "id"],
            "pk_types": ["text", "bigint"],
            "processor": "concat",
            "kind": "sql",
            "fn_sql": "public.concat_two",
            "input_types": ["text", null],
            "takes_params": true,
            "config": {},
            "effective_params": {"sep": " "},
            "strict": true,
            "max_attempts": 3,
            "result_is_array": false
        }))
        .unwrap()
    }

    #[test]
    fn where_clause_casts_each_key_part() {
        assert_eq!(
            spec().where_clause(1),
            "tenant = ($1::text[])[1]::text AND id = ($1::text[])[2]::bigint"
        );
    }

    #[test]
    fn sql_writeback_passes_params_and_skips_polymorphic_casts() {
        assert_eq!(
            spec().sql_writeback().unwrap(),
            "UPDATE public.docs SET \"Summary\" = public.concat_two(title::text, body, $2::jsonb) \
             WHERE tenant = ($1::text[])[1]::text AND id = ($1::text[])[2]::bigint"
        );
    }

    #[test]
    fn batch_statements() {
        let s = spec();
        assert_eq!(
            s.keys_source(1, 2),
            "(SELECT ev_i, ($1::text[])[(ev_i - 1) * 2 + 1 : ev_i * 2] AS ev_key FROM generate_series(1, $2) AS ev_i)"
        );
        let lock = s.lock_rows_batch_sql();
        assert!(lock.contains("t.tenant = k.ev_key[1]::text AND t.id = k.ev_key[2]::bigint"), "{lock}");
        assert!(lock.ends_with("ORDER BY t.tenant, t.id FOR UPDATE OF t SKIP LOCKED"), "{lock}");
        let update = s.sql_batch_writeback().unwrap();
        assert!(update.starts_with(
            "UPDATE public.docs t SET \"Summary\" = public.concat_two(t.title::text, t.body, $3::jsonb) FROM"
        ), "{update}");
        assert!(update.ends_with("AND NOT (t.title IS NULL OR t.body IS NULL) RETURNING k.ev_i"), "{update}");
        let json = s.json_batch_writeback();
        assert!(json.contains("(jsonb_populate_record(NULL::public.docs, k.ev_val)).\"Summary\""), "{json}");
        assert!(json.contains("$3::jsonb -> (ev_i - 1) AS ev_val"), "{json}");
    }

    #[test]
    fn json_writeback_and_reads() {
        let s = spec();
        assert!(s.json_writeback().contains("jsonb_populate_record(NULL::public.docs, $2::jsonb)).\"Summary\""));
        assert!(s.read_inputs_sql().starts_with("SELECT jsonb_build_array(title, body) FROM public.docs"));
        assert!(s.inputs_null_sql().starts_with("SELECT title IS NULL OR body IS NULL FROM"));
        assert!(s.lock_row_sql().ends_with("FOR UPDATE"));
    }
}
