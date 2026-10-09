use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpConfig {
    pub url: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    pub result: String,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub max_batch: Option<usize>,
}

fn default_method() -> String {
    "POST".into()
}

enum Placeholder {
    Input(usize),
    Inputs(usize),
    Param(String),
    Params,
    Secret(String),
}

fn placeholder(text: &str) -> Option<Result<Placeholder, String>> {
    let inner = text.strip_prefix("{{")?.strip_suffix("}}")?.trim();
    let index = |s: &str| s.parse::<usize>().map_err(|_| format!("invalid placeholder index in {text}"));
    Some(match inner.split_once('.') {
        None if inner == "input" => Ok(Placeholder::Input(0)),
        None if inner == "inputs" => Ok(Placeholder::Inputs(0)),
        None if inner == "params" => Ok(Placeholder::Params),
        Some(("input", n)) => index(n).map(Placeholder::Input),
        Some(("inputs", n)) => index(n).map(Placeholder::Inputs),
        Some(("param", name)) if !name.is_empty() => Ok(Placeholder::Param(name.to_string())),
        Some(("secret", name)) if !name.is_empty() => Ok(Placeholder::Secret(name.to_string())),
        _ => Err(format!("unknown placeholder {text}")),
    })
}

fn visit_placeholders(template: &Value, found: &mut Vec<Result<Placeholder, String>>) {
    match template {
        Value::String(s) => {
            if let Some(p) = placeholder(s) {
                found.push(p);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| visit_placeholders(v, found)),
        Value::Object(map) => map.values().for_each(|v| visit_placeholders(v, found)),
        _ => {}
    }
}

fn secret_refs_in(text: &str, out: &mut Vec<String>) {
    let mut rest = text;
    while let Some(start) = rest.find("{{secret.") {
        let after = &rest[start + "{{secret.".len()..];
        match after.find("}}") {
            Some(end) => {
                out.push(after[..end].trim().to_string());
                rest = &after[end + 2..];
            }
            None => break,
        }
    }
}

fn collect_secret_refs(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) => secret_refs_in(s, out),
        Value::Array(items) => items.iter().for_each(|v| collect_secret_refs(v, out)),
        Value::Object(map) => map.values().for_each(|v| collect_secret_refs(v, out)),
        _ => {}
    }
}

pub fn substitute_secrets(text: &str, secrets: &BTreeMap<String, String>) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("{{secret.") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "{{secret.".len()..];
        let end = after.find("}}").ok_or_else(|| "unterminated {{secret.…}} placeholder".to_string())?;
        let name = after[..end].trim();
        out.push_str(secrets.get(name).ok_or_else(|| format!("secret \"{name}\" is not set (eventual.set_secret)"))?);
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

impl HttpConfig {
    pub fn secret_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.headers.values().for_each(|v| secret_refs_in(v, &mut out));
        collect_secret_refs(&self.body, &mut out);
        out.sort();
        out.dedup();
        out
    }

    pub fn resolved_headers(&self, secrets: &BTreeMap<String, String>) -> Result<BTreeMap<String, String>, String> {
        self.headers
            .iter()
            .map(|(k, v)| Ok((k.clone(), substitute_secrets(v, secrets)?)))
            .collect()
    }

    pub fn from_json(value: &Value) -> Result<Self, String> {
        let cfg: HttpConfig = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        if !(cfg.url.starts_with("http://") || cfg.url.starts_with("https://")) {
            return Err("\"url\" must start with http:// or https://".into());
        }
        if !matches!(cfg.method.as_str(), "POST" | "PUT") {
            return Err("\"method\" must be POST or PUT".into());
        }
        if !cfg.result.is_empty() && !cfg.result.starts_with('/') {
            return Err("\"result\" must be a JSON pointer such as /data/0/embedding".into());
        }
        if cfg.max_batch == Some(0) {
            return Err("\"max_batch\" must be at least 1".into());
        }
        let stars = cfg.result.split('/').filter(|s| *s == "*").count();
        if cfg.max_batch.is_some() && stars != 1 {
            return Err("with \"max_batch\", \"result\" needs exactly one * segment (e.g. /data/*/embedding)".into());
        }
        if cfg.max_batch.is_none() && stars != 0 {
            return Err("\"result\" may only contain * when \"max_batch\" is set".into());
        }
        Ok(cfg)
    }

    pub fn check(&self, inputs: usize, defaults: &Value) -> Result<(), String> {
        let defaults = defaults.as_object().ok_or("processor defaults must be a JSON object")?;
        let mut found = Vec::new();
        visit_placeholders(&self.body, &mut found);
        let mut uses_input = false;
        for p in found {
            match p? {
                Placeholder::Input(n) | Placeholder::Inputs(n) if n >= inputs => {
                    return Err(format!("placeholder refers to input {n}, but the processor has {inputs} input(s)"));
                }
                Placeholder::Input(_) if self.max_batch.is_some() => {
                    return Err("batched requests use {{inputs.N}} (arrays), not {{input.N}}".into());
                }
                Placeholder::Inputs(_) if self.max_batch.is_none() => {
                    return Err("{{inputs.N}} needs \"max_batch\"; use {{input.N}} for single requests".into());
                }
                Placeholder::Secret(name) if name.is_empty() => return Err("empty secret name".into()),
                Placeholder::Param(name) if !defaults.contains_key(&name) => {
                    return Err(format!("placeholder refers to parameter \"{name}\", which has no default"));
                }
                Placeholder::Input(_) | Placeholder::Inputs(_) => uses_input = true,
                _ => {}
            }
        }
        if !uses_input {
            return Err("the body template does not use any input placeholder".into());
        }
        Ok(())
    }
}

fn expand(
    template: &Value,
    rows: &[Vec<Value>],
    params: &Value,
    secrets: &BTreeMap<String, String>,
) -> Result<Value, String> {
    Ok(match template {
        Value::String(s) => match placeholder(s) {
            None => template.clone(),
            Some(Ok(Placeholder::Secret(name))) => Value::String(
                secrets
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| format!("secret \"{name}\" is not set (eventual.set_secret)"))?,
            ),
            Some(p) => match p? {
                Placeholder::Input(n) => rows
                    .first()
                    .and_then(|r| r.get(n))
                    .cloned()
                    .ok_or_else(|| format!("missing input {n}"))?,
                Placeholder::Inputs(n) => Value::Array(
                    rows.iter()
                        .map(|r| r.get(n).cloned().ok_or_else(|| format!("missing input {n}")))
                        .collect::<Result<_, _>>()?,
                ),
                Placeholder::Param(name) => {
                    params.get(&name).cloned().ok_or_else(|| format!("missing parameter \"{name}\""))?
                }
                Placeholder::Params => params.clone(),
                Placeholder::Secret(_) => unreachable!("handled above"),
            },
        },
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| expand(v, rows, params, secrets)).collect::<Result<_, _>>()?)
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| Ok((k.clone(), expand(v, rows, params, secrets)?)))
                .collect::<Result<Map<_, _>, String>>()?,
        ),
        other => other.clone(),
    })
}

pub fn build_body(
    cfg: &HttpConfig,
    inputs: &[Value],
    params: &Value,
    secrets: &BTreeMap<String, String>,
) -> Result<Value, String> {
    expand(&cfg.body, &[inputs.to_vec()], params, secrets)
}

pub fn build_batch_body(
    cfg: &HttpConfig,
    rows: &[Vec<Value>],
    params: &Value,
    secrets: &BTreeMap<String, String>,
) -> Result<Value, String> {
    expand(&cfg.body, rows, params, secrets)
}

fn check_value(value: Value, result_is_array: bool) -> Result<Value, String> {
    if result_is_array && !value.is_array() {
        return Err(format!("expected a JSON array for an array column, got {}", truncate(&value)));
    }
    if let Value::Array(items) = &value {
        if items.iter().any(|v| v.is_number() && !v.as_f64().map(f64::is_finite).unwrap_or(false)) {
            return Err("result contains a non-finite number".into());
        }
    }
    if value.is_null() {
        return Err("result is null".into());
    }
    Ok(value)
}

fn truncate(value: &Value) -> String {
    let text = value.to_string();
    if text.len() > 200 {
        format!("{}...", &text[..200])
    } else {
        text
    }
}

pub fn extract(cfg: &HttpConfig, response: &Value, result_is_array: bool) -> Result<Value, String> {
    let value = response
        .pointer(&cfg.result)
        .cloned()
        .ok_or_else(|| format!("result pointer {} not found in response {}", cfg.result, truncate(response)))?;
    check_value(value, result_is_array)
}

pub fn extract_batch(cfg: &HttpConfig, response: &Value, rows: usize, result_is_array: bool) -> Result<Vec<Value>, String> {
    let (before, after) = cfg.result.split_once("/*").ok_or("result pointer has no * segment")?;
    let list = response
        .pointer(before)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{before} is not an array in response {}", truncate(response)))?;
    if list.len() != rows {
        return Err(format!("response has {} results for {rows} rows", list.len()));
    }
    list.iter()
        .map(|item| {
            let value = if after.is_empty() { Some(item.clone()) } else { item.pointer(after).cloned() };
            let value = value.ok_or_else(|| format!("result pointer {} not found in {}", cfg.result, truncate(item)))?;
            check_value(value, result_is_array)
        })
        .collect()
}

pub fn send_json(
    method: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
    body: &Value,
    timeout: Duration,
) -> Result<Value, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into();
    let mut request = match method {
        "PUT" => agent.put(url),
        _ => agent.post(url),
    };
    for (name, value) in headers {
        request = request.header(name, value);
    }
    let mut response = request.send_json(body).map_err(|e| format!("request to {url} failed: {e}"))?;
    let status = response.status();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("reading response from {url} failed: {e}"))?;
    let parsed: Result<Value, _> = serde_json::from_str(&text);
    if !status.is_success() {
        let detail = parsed
            .ok()
            .and_then(|v| match v.get("error") {
                Some(Value::String(s)) => Some(s.clone()),
                Some(other) => Some(other.to_string()),
                None => None,
            })
            .unwrap_or_else(|| text.chars().take(500).collect());
        return Err(format!("HTTP {} from {url}: {detail}", status.as_u16()));
    }
    parsed.map_err(|e| format!("invalid JSON from {url}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(value: Value) -> HttpConfig {
        HttpConfig::from_json(&value).unwrap()
    }

    #[test]
    fn templates_expand_with_types_preserved() {
        let c = cfg(json!({"url": "http://x/embed", "body": {"model": "m", "input": "{{input}}",
            "second": "{{input.1}}", "opts": "{{params}}", "dims": "{{param.dims}}", "keep": "{{not a placeholder"},
            "result": "/data/0/embedding"}));
        c.check(2, &json!({"dims": 8})).unwrap();
        let body = build_body(&c, &[json!("hello"), json!([1, 2])], &json!({"dims": 8}), &BTreeMap::new()).unwrap();
        assert_eq!(body, json!({"model": "m", "input": "hello", "second": [1, 2], "opts": {"dims": 8}, "dims": 8,
                                "keep": "{{not a placeholder"}));
    }

    #[test]
    fn configuration_errors() {
        assert!(HttpConfig::from_json(&json!({"url": "ftp://x", "body": {}, "result": "/a"})).is_err());
        assert!(HttpConfig::from_json(&json!({"url": "http://x", "body": {}, "result": "a"})).is_err());
        assert!(HttpConfig::from_json(&json!({"url": "http://x", "body": {}, "result": "/a/*/b"})).is_err());
        assert!(HttpConfig::from_json(&json!({"url": "http://x", "body": {}, "result": "/a", "max_batch": 2})).is_err());
        let c = cfg(json!({"url": "http://x", "body": {"a": "{{input.3}}"}, "result": "/r"}));
        assert!(c.check(1, &json!({})).unwrap_err().contains("input 3"));
        let c = cfg(json!({"url": "http://x", "body": {"a": "{{param.k}}"}, "result": "/r"}));
        assert!(c.check(1, &json!({})).unwrap_err().contains("parameter \"k\""));
        let c = cfg(json!({"url": "http://x", "body": {"a": "{{wat}}"}, "result": "/r"}));
        assert!(c.check(1, &json!({})).unwrap_err().contains("unknown placeholder"));
        let c = cfg(json!({"url": "http://x", "body": {"a": "{{input}}"}, "result": "/r/*", "max_batch": 4}));
        assert!(c.check(1, &json!({})).unwrap_err().contains("{{inputs.N}}"));
        let c = cfg(json!({"url": "http://x", "body": {"a": 1}, "result": "/r"}));
        assert!(c.check(1, &json!({})).unwrap_err().contains("does not use any input"));
    }

    #[test]
    fn secrets_in_headers_and_body() {
        let c = cfg(json!({"url": "https://api.example/v1", "headers": {"Authorization": "Bearer {{secret.key}}",
            "X-Org": "{{secret.org}}-team"}, "body": {"input": "{{input}}", "token": "{{secret.key}}"}, "result": "/r"}));
        c.check(1, &json!({})).unwrap();
        assert_eq!(c.secret_names(), vec!["key", "org"]);
        let mut secrets = BTreeMap::new();
        secrets.insert("key".to_string(), "s3cr3t".to_string());
        assert!(c.resolved_headers(&secrets).unwrap_err().contains("secret \"org\" is not set"));
        secrets.insert("org".to_string(), "acme".to_string());
        let headers = c.resolved_headers(&secrets).unwrap();
        assert_eq!(headers["Authorization"], "Bearer s3cr3t");
        assert_eq!(headers["X-Org"], "acme-team");
        let body = build_body(&c, &[json!("x")], &json!({}), &secrets).unwrap();
        assert_eq!(body, json!({"input": "x", "token": "s3cr3t"}));
    }

    #[test]
    fn extraction_single_and_batched() {
        let c = cfg(json!({"url": "http://x", "body": {"t": "{{input}}"}, "result": "/data/0/embedding"}));
        let response = json!({"data": [{"embedding": [0.5, 1.5]}]});
        assert_eq!(extract(&c, &response, true).unwrap(), json!([0.5, 1.5]));
        assert!(extract(&c, &json!({"data": []}), true).unwrap_err().contains("not found"));
        assert!(extract(&c, &json!({"data": [{"embedding": "x"}]}), true).unwrap_err().contains("JSON array"));

        let b = cfg(json!({"url": "http://x", "body": {"input": "{{inputs}}"}, "result": "/data/*/embedding", "max_batch": 8}));
        b.check(1, &json!({})).unwrap();
        let body = build_batch_body(&b, &[vec![json!("a")], vec![json!("b")]], &json!({}), &BTreeMap::new()).unwrap();
        assert_eq!(body, json!({"input": ["a", "b"]}));
        let response = json!({"data": [{"embedding": [1]}, {"embedding": [2]}]});
        assert_eq!(extract_batch(&b, &response, 2, true).unwrap(), vec![json!([1]), json!([2])]);
        assert!(extract_batch(&b, &response, 3, true).unwrap_err().contains("2 results for 3 rows"));
        let flat = cfg(json!({"url": "http://x", "body": {"x": "{{inputs.0}}"}, "result": "/out/*", "max_batch": 2}));
        assert_eq!(extract_batch(&flat, &json!({"out": ["A", "B"]}), 2, false).unwrap(), vec![json!("A"), json!("B")]);
    }
}
