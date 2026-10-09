use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    #[serde(default)]
    pub datatype: Option<String>,
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    pub name: String,
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TritonConfig {
    #[serde(default)]
    pub url: Option<String>,
    pub model: String,
    #[serde(default)]
    pub version: Option<String>,
    pub inputs: Vec<TensorSpec>,
    pub output: OutputSpec,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default = "default_true")]
    pub send_parameters: bool,
    #[serde(default)]
    pub parameter_tensors: BTreeMap<String, ParamTensor>,
    #[serde(default)]
    pub max_batch: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamTensor {
    pub name: String,
    pub datatype: String,
}

fn default_true() -> bool {
    true
}

const DATATYPES: &[&str] = &[
    "BOOL", "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64", "FP16", "FP32",
    "FP64", "BYTES",
];

impl TritonConfig {
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let cfg: TritonConfig = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        if cfg.model.is_empty() {
            return Err("\"model\" must not be empty".into());
        }
        if cfg.inputs.is_empty() {
            return Err("\"inputs\" must list at least one tensor".into());
        }
        for spec in &cfg.inputs {
            if let Some(dt) = &spec.datatype {
                if !DATATYPES.contains(&dt.as_str()) {
                    return Err(format!("unsupported datatype \"{dt}\" for input \"{}\"", spec.name));
                }
            }
        }
        for (param, spec) in &cfg.parameter_tensors {
            if !DATATYPES.contains(&spec.datatype.as_str()) {
                return Err(format!("unsupported datatype \"{}\" for parameter \"{param}\"", spec.datatype));
            }
        }
        if cfg.max_batch == Some(0) {
            return Err("\"max_batch\" must be at least 1".into());
        }
        Ok(cfg)
    }

    pub fn check_parameters(&self, defaults: &Value) -> Result<(), String> {
        let defaults = defaults.as_object().ok_or("processor defaults must be a JSON object")?;
        for param in self.parameter_tensors.keys() {
            if !defaults.contains_key(param) {
                return Err(format!("parameter tensor \"{param}\" has no default in the processor defaults"));
            }
        }
        if !self.send_parameters {
            for param in defaults.keys() {
                if !self.parameter_tensors.contains_key(param) {
                    return Err(format!(
                        "parameter \"{param}\" would not reach the model: map it in \"parameter_tensors\" or enable \"send_parameters\""
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn infer_path(&self) -> String {
        match &self.version {
            Some(version) if !version.is_empty() => {
                format!("/v2/models/{}/versions/{}/infer", self.model, version)
            }
            _ => format!("/v2/models/{}/infer", self.model),
        }
    }
}

fn flatten(value: &Value, shape: &mut Vec<i64>, depth: usize, out: &mut Vec<Value>) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            if shape.len() == depth {
                shape.push(items.len() as i64);
            } else if shape[depth] != items.len() as i64 {
                return Err("input array is not rectangular".into());
            }
            for item in items {
                flatten(item, shape, depth + 1, out)?;
            }
            Ok(())
        }
        Value::Null => Err("input array contains NULL".into()),
        leaf => {
            if depth < shape.len() {
                return Err("input array is not rectangular".into());
            }
            out.push(leaf.clone());
            Ok(())
        }
    }
}

fn default_datatype(sample: Option<&Value>) -> &'static str {
    match sample {
        Some(Value::String(_)) => "BYTES",
        Some(Value::Bool(_)) => "BOOL",
        _ => "FP32",
    }
}

pub fn input_tensor(spec: &TensorSpec, value: &Value) -> Result<Value, String> {
    let mut shape = Vec::new();
    let mut data = Vec::new();
    match value {
        Value::Null => return Err(format!("input \"{}\" is NULL", spec.name)),
        Value::Array(_) => flatten(value, &mut shape, 0, &mut data)?,
        Value::Object(_) => return Err(format!("input \"{}\": JSON objects cannot be sent as tensors", spec.name)),
        leaf => {
            shape.push(1);
            data.push(leaf.clone());
        }
    }
    let datatype = spec
        .datatype
        .clone()
        .unwrap_or_else(|| default_datatype(data.first()).to_string());
    for item in &data {
        let ok = match datatype.as_str() {
            "BYTES" => item.is_string(),
            "BOOL" => item.is_boolean(),
            _ => item.is_number(),
        };
        if !ok {
            return Err(format!("input \"{}\": value {item} does not match datatype {datatype}", spec.name));
        }
    }
    if let Some(declared) = &spec.shape {
        let declared_len: i64 = declared.iter().product();
        if declared.iter().any(|d| *d < 0) || declared_len != data.len() as i64 {
            return Err(format!(
                "input \"{}\" has {} elements but the configured shape {:?} needs {}",
                spec.name,
                data.len(),
                declared,
                declared_len
            ));
        }
        shape = declared.clone();
    }
    Ok(json!({"name": spec.name, "shape": shape, "datatype": datatype, "data": data}))
}

fn parameter_tensors_and_rest(cfg: &TritonConfig, params: &Value) -> Result<(Vec<Value>, Map<String, Value>), String> {
    let empty = Map::new();
    let params = params.as_object().unwrap_or(&empty);
    let mut tensors = Vec::new();
    for (param, spec) in &cfg.parameter_tensors {
        let value = params
            .get(param)
            .ok_or_else(|| format!("parameter \"{param}\" has no value"))?;
        let tensor_spec = TensorSpec { name: spec.name.clone(), datatype: Some(spec.datatype.clone()), shape: None };
        tensors.push(input_tensor(&tensor_spec, value)?);
    }
    let rest = params
        .iter()
        .filter(|(k, _)| !cfg.parameter_tensors.contains_key(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok((tensors, rest))
}

fn finish_request(cfg: &TritonConfig, mut tensors: Vec<Value>, params: &Value) -> Result<Value, String> {
    let (param_tensors, rest) = parameter_tensors_and_rest(cfg, params)?;
    tensors.extend(param_tensors);
    let mut request = Map::new();
    request.insert("inputs".into(), Value::Array(tensors));
    request.insert("outputs".into(), json!([{"name": cfg.output.name}]));
    if cfg.send_parameters && !rest.is_empty() {
        request.insert("parameters".into(), Value::Object(rest));
    }
    Ok(Value::Object(request))
}

pub fn build_batch_request(cfg: &TritonConfig, rows: &[Vec<Value>], params: &Value) -> Result<Value, String> {
    if rows.is_empty() {
        return Err("empty batch".into());
    }
    let mut tensors = Vec::new();
    for (i, spec) in cfg.inputs.iter().enumerate() {
        let mut item_shape: Option<Vec<i64>> = None;
        let mut datatype = String::new();
        let mut data = Vec::new();
        for row in rows {
            let value = row.get(i).ok_or_else(|| format!("expected {} inputs per row", cfg.inputs.len()))?;
            let tensor = input_tensor(spec, value)?;
            let shape: Vec<i64> = serde_json::from_value(tensor["shape"].clone()).map_err(|e| e.to_string())?;
            match &item_shape {
                None => {
                    item_shape = Some(shape);
                    datatype = tensor["datatype"].as_str().unwrap_or_default().to_string();
                }
                Some(first) if *first != shape => {
                    return Err(format!("input \"{}\" has different shapes within one batch", spec.name));
                }
                _ => {}
            }
            data.extend(tensor["data"].as_array().cloned().unwrap_or_default());
        }
        let item_shape = item_shape.unwrap_or_default();
        let mut shape = vec![rows.len() as i64];
        if item_shape != [1] {
            shape.extend(item_shape);
        }
        tensors.push(json!({"name": spec.name, "shape": shape, "datatype": datatype, "data": data}));
    }
    finish_request(cfg, tensors, params)
}

pub fn build_request(cfg: &TritonConfig, inputs: &[Value], params: &Value) -> Result<Value, String> {
    if inputs.len() != cfg.inputs.len() {
        return Err(format!("expected {} inputs, got {}", cfg.inputs.len(), inputs.len()));
    }
    let tensors = cfg
        .inputs
        .iter()
        .zip(inputs)
        .map(|(spec, value)| input_tensor(spec, value))
        .collect::<Result<Vec<_>, _>>()?;
    finish_request(cfg, tensors, params)
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub value: Value,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
}

fn reshape(data: &[Value], dims: &[i64]) -> Value {
    if dims.len() <= 1 {
        return Value::Array(data.to_vec());
    }
    let chunk = (data.len() as i64 / dims[0].max(1)) as usize;
    Value::Array(data.chunks(chunk.max(1)).map(|part| reshape(part, &dims[1..])).collect())
}

struct CheckedOutput {
    shape: Vec<i64>,
    data: Vec<Value>,
    model_name: Option<String>,
    model_version: Option<String>,
}

pub fn parse_response(cfg: &TritonConfig, response: &Value, result_is_array: bool) -> Result<InferenceResult, String> {
    let out = check_output(cfg, response)?;
    let value = if result_is_array {
        let mut dims = out.shape.clone();
        while dims.len() > 1 && dims[0] == 1 {
            dims.remove(0);
        }
        reshape(&out.data, &dims)
    } else {
        if out.data.len() != 1 {
            return Err(format!("expected a single output value for a scalar column, got {}", out.data.len()));
        }
        out.data[0].clone()
    };
    Ok(InferenceResult { value, model_name: out.model_name, model_version: out.model_version })
}

pub fn parse_batch_response(
    cfg: &TritonConfig,
    response: &Value,
    rows: usize,
    result_is_array: bool,
) -> Result<Vec<InferenceResult>, String> {
    let out = check_output(cfg, response)?;
    if out.shape.first().copied() != Some(rows as i64) {
        return Err(format!("batched output shape {:?} does not start with the batch size {rows}", out.shape));
    }
    let item_dims = out.shape[1..].to_vec();
    let per_item = out.data.len() / rows.max(1);
    out.data
        .chunks(per_item.max(1))
        .map(|chunk| {
            let value = if result_is_array {
                reshape(chunk, if item_dims.is_empty() { &[1] } else { &item_dims })
            } else if chunk.len() == 1 {
                chunk[0].clone()
            } else {
                return Err(format!("expected a single output value per row for a scalar column, got {}", chunk.len()));
            };
            Ok(InferenceResult { value, model_name: out.model_name.clone(), model_version: out.model_version.clone() })
        })
        .collect()
}

fn check_output(cfg: &TritonConfig, response: &Value) -> Result<CheckedOutput, String> {
    if let Some(error) = response.get("error").and_then(Value::as_str) {
        return Err(format!("triton error: {error}"));
    }
    let model_name = response.get("model_name").and_then(Value::as_str).map(str::to_owned);
    let model_version = response.get("model_version").and_then(Value::as_str).map(str::to_owned);
    if let Some(name) = &model_name {
        if name != &cfg.model {
            return Err(format!("response came from model \"{name}\", expected \"{}\"", cfg.model));
        }
    }
    if let (Some(expected), Some(actual)) = (cfg.version.as_ref(), model_version.as_ref()) {
        if !expected.is_empty() && expected != actual {
            return Err(format!("response came from model version {actual}, expected {expected}"));
        }
    }
    let outputs = response
        .get("outputs")
        .and_then(Value::as_array)
        .ok_or("response has no outputs")?;
    let output = outputs
        .iter()
        .find(|o| o.get("name").and_then(Value::as_str) == Some(cfg.output.name.as_str()))
        .ok_or_else(|| format!("response has no output named \"{}\"", cfg.output.name))?;
    let datatype = output.get("datatype").and_then(Value::as_str).unwrap_or("");
    let shape: Vec<i64> = output
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("output has no shape")?
        .iter()
        .map(|d| d.as_i64().ok_or("output shape is not numeric"))
        .collect::<Result<_, _>>()?;
    let data = output
        .get("data")
        .and_then(Value::as_array)
        .ok_or("output has no JSON data (binary outputs are not supported)")?;
    let expected_len: i64 = shape.iter().product();
    if expected_len != data.len() as i64 {
        return Err(format!("output shape {shape:?} does not match {} data elements", data.len()));
    }
    if let Some(expected) = &cfg.output.shape {
        let matches = expected.len() == shape.len()
            && expected.iter().zip(&shape).all(|(e, a)| *e < 0 || e == a);
        if !matches {
            return Err(format!("output shape {shape:?} does not match expected {expected:?}"));
        }
    }
    for item in data {
        let ok = match datatype {
            "BYTES" => item.is_string(),
            "BOOL" => item.is_boolean(),
            _ => item.as_f64().map(f64::is_finite).unwrap_or(false),
        };
        if !ok {
            return Err(format!("output contains an invalid {datatype} value: {item}"));
        }
    }
    Ok(CheckedOutput { shape, data: data.clone(), model_name, model_version })
}

pub fn infer(base_url: &str, cfg: &TritonConfig, request: &Value, timeout: Duration) -> Result<Value, String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), cfg.infer_path());
    crate::http::send_json("POST", &url, &BTreeMap::new(), request, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine() -> TritonConfig {
        TritonConfig::from_json(&json!({
            "model": "learning_affine", "version": "1",
            "inputs": [{"name": "INPUT", "datatype": "FP32"}],
            "output": {"name": "OUTPUT", "shape": [2]}
        }))
        .unwrap()
    }

    #[test]
    fn config_validation() {
        assert!(TritonConfig::from_json(&json!({"model": "m", "inputs": [], "output": {"name": "o"}})).is_err());
        assert!(TritonConfig::from_json(&json!({"model": "m", "inputs": [{"name": "i", "datatype": "FLOAT"}], "output": {"name": "o"}})).is_err());
        assert!(TritonConfig::from_json(&json!({"model": "m", "inputs": [{"name": "i"}], "output": {"name": "o"}, "extra": 1})).is_err());
        let cfg = affine();
        assert_eq!(cfg.infer_path(), "/v2/models/learning_affine/versions/1/infer");
        assert!(cfg.send_parameters);
    }

    #[test]
    fn request_for_vector_and_text() {
        let req = build_request(&affine(), &[json!([1.0, 3.0])], &json!({})).unwrap();
        assert_eq!(
            req,
            json!({"inputs": [{"name": "INPUT", "shape": [2], "datatype": "FP32", "data": [1.0, 3.0]}],
                   "outputs": [{"name": "OUTPUT"}]})
        );
        let text = TritonConfig::from_json(&json!({
            "model": "embed", "inputs": [{"name": "TEXT"}], "output": {"name": "EMBEDDING"}
        }))
        .unwrap();
        let req = build_request(&text, &[json!("hello")], &json!({"normalize": true})).unwrap();
        assert_eq!(req["inputs"][0], json!({"name": "TEXT", "shape": [1], "datatype": "BYTES", "data": ["hello"]}));
        assert_eq!(req["parameters"], json!({"normalize": true}));
    }

    #[test]
    fn parameters_as_tensors() {
        let cfg = TritonConfig::from_json(&json!({
            "model": "embed", "inputs": [{"name": "TEXT"}], "output": {"name": "E"},
            "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
            "send_parameters": false
        }))
        .unwrap();
        let req = build_request(&cfg, &[json!("hi")], &json!({"normalize": true})).unwrap();
        assert_eq!(req["inputs"][1], json!({"name": "NORMALIZE", "shape": [1], "datatype": "BOOL", "data": [true]}));
        assert!(req.get("parameters").is_none());
        assert!(cfg.check_parameters(&json!({"normalize": false})).is_ok());
        let err = cfg.check_parameters(&json!({"normalize": false, "pooling": "cls"})).unwrap_err();
        assert!(err.contains("\"pooling\" would not reach the model"), "{err}");
        assert!(cfg.check_parameters(&json!({})).unwrap_err().contains("no default"));
        assert!(build_request(&cfg, &[json!("hi")], &json!({})).unwrap_err().contains("no value"));
        let both = TritonConfig::from_json(&json!({
            "model": "m", "inputs": [{"name": "T"}], "output": {"name": "E"},
            "parameter_tensors": {"k": {"name": "K", "datatype": "INT64"}}
        }))
        .unwrap();
        let req = build_request(&both, &[json!("x")], &json!({"k": 3, "mode": "fast"})).unwrap();
        assert_eq!(req["parameters"], json!({"mode": "fast"}));
        assert_eq!(req["inputs"][1]["datatype"], json!("INT64"));
    }

    #[test]
    fn batched_requests_and_responses() {
        let cfg = TritonConfig::from_json(&json!({
            "model": "embed", "inputs": [{"name": "TEXT"}], "output": {"name": "E", "shape": [-1, 2]},
            "parameter_tensors": {"normalize": {"name": "N", "datatype": "BOOL"}}, "max_batch": 8
        }))
        .unwrap();
        let rows = vec![vec![json!("a")], vec![json!("b")], vec![json!("c")]];
        let req = build_batch_request(&cfg, &rows, &json!({"normalize": true})).unwrap();
        assert_eq!(req["inputs"][0], json!({"name": "TEXT", "shape": [3], "datatype": "BYTES", "data": ["a", "b", "c"]}));
        assert_eq!(req["inputs"][1]["shape"], json!([1]));
        let response = json!({"model_name": "embed", "outputs": [
            {"name": "E", "datatype": "FP32", "shape": [3, 2], "data": [1, 2, 3, 4, 5, 6]}]});
        let out = parse_batch_response(&cfg, &response, 3, true).unwrap();
        assert_eq!(out.iter().map(|r| r.value.clone()).collect::<Vec<_>>(), vec![json!([1, 2]), json!([3, 4]), json!([5, 6])]);
        assert!(parse_batch_response(&cfg, &response, 2, true).unwrap_err().contains("batch size 2"));
        let vec_cfg = TritonConfig::from_json(&json!({
            "model": "m", "inputs": [{"name": "X", "datatype": "FP32"}], "output": {"name": "Y"}, "max_batch": 4
        }))
        .unwrap();
        let req = build_batch_request(&vec_cfg, &[vec![json!([1, 2])], vec![json!([3, 4])]], &json!({})).unwrap();
        assert_eq!(req["inputs"][0]["shape"], json!([2, 2]));
        assert!(build_batch_request(&vec_cfg, &[vec![json!([1, 2])], vec![json!([3])]], &json!({}))
            .unwrap_err()
            .contains("different shapes"));
        let scalar = json!({"outputs": [{"name": "Y", "datatype": "FP32", "shape": [2], "data": [7, 8]}]});
        let out = parse_batch_response(&vec_cfg, &scalar, 2, false).unwrap();
        assert_eq!(out[1].value, json!(8));
        assert!(TritonConfig::from_json(&json!({"model": "m", "inputs": [{"name": "X"}], "output": {"name": "Y"}, "max_batch": 0})).is_err());
    }

    #[test]
    fn request_rejects_bad_inputs() {
        let cfg = affine();
        assert!(build_request(&cfg, &[json!(null)], &json!({})).is_err());
        assert!(build_request(&cfg, &[json!(["a", "b"])], &json!({})).unwrap_err().contains("does not match datatype"));
        assert!(build_request(&cfg, &[json!([[1], [2, 3]])], &json!({})).unwrap_err().contains("rectangular"));
        assert!(build_request(&cfg, &[], &json!({})).is_err());
        let shaped = TritonConfig::from_json(&json!({
            "model": "m", "inputs": [{"name": "I", "shape": [1, 2]}], "output": {"name": "O"}
        }))
        .unwrap();
        let req = build_request(&shaped, &[json!([1, 2])], &json!({})).unwrap();
        assert_eq!(req["inputs"][0]["shape"], json!([1, 2]));
        assert!(build_request(&shaped, &[json!([1, 2, 3])], &json!({})).is_err());
    }

    #[test]
    fn response_parsing() {
        let cfg = affine();
        let response = json!({"model_name": "learning_affine", "model_version": "1",
            "outputs": [{"name": "OUTPUT", "datatype": "FP32", "shape": [2], "data": [3.0, 7.0]}]});
        let result = parse_response(&cfg, &response, true).unwrap();
        assert_eq!(result.value, json!([3.0, 7.0]));
        assert_eq!(result.model_version.as_deref(), Some("1"));
        assert!(parse_response(&cfg, &response, false).unwrap_err().contains("single output"));
    }

    #[test]
    fn response_validation_failures() {
        let cfg = affine();
        let wrong_model = json!({"model_name": "other", "outputs": []});
        assert!(parse_response(&cfg, &wrong_model, true).unwrap_err().contains("expected \"learning_affine\""));
        let wrong_shape = json!({"outputs": [{"name": "OUTPUT", "datatype": "FP32", "shape": [3], "data": [1, 2, 3]}]});
        assert!(parse_response(&cfg, &wrong_shape, true).unwrap_err().contains("expected [2]"));
        let missing = json!({"outputs": [{"name": "OTHER", "datatype": "FP32", "shape": [2], "data": [1, 2]}]});
        assert!(parse_response(&cfg, &missing, true).unwrap_err().contains("no output named"));
        let mismatch = json!({"outputs": [{"name": "OUTPUT", "datatype": "FP32", "shape": [2], "data": [1]}]});
        assert!(parse_response(&cfg, &mismatch, true).is_err());
        let error = json!({"error": "boom"});
        assert!(parse_response(&cfg, &error, true).unwrap_err().contains("boom"));
        let bad_value = json!({"outputs": [{"name": "OUTPUT", "datatype": "FP32", "shape": [2], "data": [1, "x"]}]});
        assert!(parse_response(&cfg, &bad_value, true).unwrap_err().contains("invalid FP32"));
    }

    #[test]
    fn batch_dimension_is_dropped_and_matrices_nest() {
        let cfg = TritonConfig::from_json(&json!({"model": "m", "inputs": [{"name": "I"}], "output": {"name": "O"}})).unwrap();
        let batched = json!({"outputs": [{"name": "O", "datatype": "FP32", "shape": [1, 3], "data": [1, 2, 3]}]});
        assert_eq!(parse_response(&cfg, &batched, true).unwrap().value, json!([1, 2, 3]));
        let matrix = json!({"outputs": [{"name": "O", "datatype": "FP32", "shape": [2, 2], "data": [1, 2, 3, 4]}]});
        assert_eq!(parse_response(&cfg, &matrix, true).unwrap().value, json!([[1, 2], [3, 4]]));
        let scalar = json!({"outputs": [{"name": "O", "datatype": "BYTES", "shape": [1], "data": ["hi"]}]});
        assert_eq!(parse_response(&cfg, &scalar, false).unwrap().value, json!("hi"));
    }
}
