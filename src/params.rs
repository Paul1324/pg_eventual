use serde_json::{Map, Value};

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub fn merge(defaults: &Value, overrides: &Value) -> Result<Value, String> {
    let defaults = defaults
        .as_object()
        .ok_or("processor defaults must be a JSON object")?;
    let overrides = match overrides {
        Value::Null => return Ok(Value::Object(defaults.clone())),
        Value::Object(map) => map,
        _ => return Err("column parameters must be a JSON object".into()),
    };
    let mut effective: Map<String, Value> = defaults.clone();
    for (key, value) in overrides {
        let Some(default) = defaults.get(key) else {
            let mut known: Vec<&str> = defaults.keys().map(String::as_str).collect();
            known.sort_unstable();
            let known = if known.is_empty() { "none".to_string() } else { known.join(", ") };
            return Err(format!("unknown processor parameter \"{key}\" (supported: {known})"));
        };
        if !default.is_null() && json_kind(default) != json_kind(value) {
            return Err(format!(
                "processor parameter \"{key}\" must be a {}, got {}",
                json_kind(default),
                json_kind(value)
            ));
        }
        effective.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(effective))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn overrides_replace_defaults() {
        let out = merge(&json!({"normalize": false, "k": 3}), &json!({"normalize": true})).unwrap();
        assert_eq!(out, json!({"normalize": true, "k": 3}));
    }

    #[test]
    fn null_overrides_mean_defaults() {
        assert_eq!(merge(&json!({"a": 1}), &Value::Null).unwrap(), json!({"a": 1}));
        assert_eq!(merge(&json!({}), &json!({})).unwrap(), json!({}));
    }

    #[test]
    fn unknown_and_mistyped_parameters_fail() {
        let err = merge(&json!({"normalize": false}), &json!({"normalise": true})).unwrap_err();
        assert!(err.contains("unknown processor parameter \"normalise\""), "{err}");
        assert!(err.contains("normalize"));
        let err = merge(&json!({"normalize": false}), &json!({"normalize": "yes"})).unwrap_err();
        assert!(err.contains("must be a boolean"), "{err}");
        assert!(merge(&json!({}), &json!([1])).is_err());
        assert!(merge(&json!([]), &json!({})).is_err());
    }

    #[test]
    fn null_default_accepts_any_type() {
        assert_eq!(merge(&json!({"x": null}), &json!({"x": [1, 2]})).unwrap(), json!({"x": [1, 2]}));
    }
}
