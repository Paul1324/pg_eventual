use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum ModArg {
    Ident(String),
    Str(String),
    Int(i64),
    Float(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub processor: String,
    pub inputs: Vec<String>,
    pub params: Map<String, Value>,
}

fn describe(arg: &ModArg) -> String {
    match arg {
        ModArg::Ident(v) => format!("identifier {v}"),
        ModArg::Str(v) => format!("string '{v}'"),
        ModArg::Int(v) => format!("number {v}"),
        ModArg::Float(v) => format!("number {v}"),
        ModArg::Bool(v) => format!("boolean {v}"),
    }
}

fn looks_like_json_object(text: &str) -> bool {
    text.trim_start().starts_with('{')
}

pub fn parse(args: &[ModArg]) -> Result<Declaration, String> {
    let usage = "expected eventual.derived(processor, input_column [, ...] [, '{\"param\": value}'])";
    let (first, rest) = args
        .split_first()
        .ok_or_else(|| format!("missing processor name: {usage}"))?;
    let processor = match first {
        ModArg::Ident(name) => name.clone(),
        ModArg::Str(name) if !looks_like_json_object(name) && !name.is_empty() => name.clone(),
        other => return Err(format!("the first argument must name a processor, got {}: {usage}", describe(other))),
    };

    let mut params = Map::new();
    let mut inputs = Vec::new();
    for (position, arg) in rest.iter().enumerate() {
        let is_last = position + 1 == rest.len();
        match arg {
            ModArg::Ident(name) => inputs.push(name.clone()),
            ModArg::Str(text) if looks_like_json_object(text) => {
                if !is_last {
                    return Err(format!("the parameter object must be the last argument: {usage}"));
                }
                match serde_json::from_str::<Value>(text) {
                    Ok(Value::Object(map)) => params = map,
                    Ok(_) => return Err("processor parameters must be a JSON object".into()),
                    Err(error) => return Err(format!("invalid JSON in processor parameters: {error}")),
                }
            }
            ModArg::Str(name) if !name.is_empty() => inputs.push(name.clone()),
            other => return Err(format!("unexpected {} in derived column declaration: {usage}", describe(other))),
        }
    }
    if inputs.is_empty() {
        return Err(format!("a derived column needs at least one input column: {usage}"));
    }
    for (i, name) in inputs.iter().enumerate() {
        if inputs[..i].contains(name) {
            return Err(format!("input column \"{name}\" is listed more than once"));
        }
    }
    Ok(Declaration { processor, inputs, params })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ident(s: &str) -> ModArg {
        ModArg::Ident(s.into())
    }
    fn string(s: &str) -> ModArg {
        ModArg::Str(s.into())
    }

    #[test]
    fn identifiers_and_params() {
        let d = parse(&[ident("embed_text"), ident("body"), string(r#"{"normalize": true}"#)]).unwrap();
        assert_eq!(d.processor, "embed_text");
        assert_eq!(d.inputs, vec!["body"]);
        assert_eq!(Value::Object(d.params), json!({"normalize": true}));
    }

    #[test]
    fn quoted_strings_are_accepted_for_names() {
        let d = parse(&[string("concat"), string("Title"), ident("body")]).unwrap();
        assert_eq!(d.processor, "concat");
        assert_eq!(d.inputs, vec!["Title", "body"]);
        assert!(d.params.is_empty());
    }

    #[test]
    fn rejects_missing_inputs_and_bad_params() {
        assert!(parse(&[]).unwrap_err().contains("missing processor"));
        assert!(parse(&[ident("p")]).unwrap_err().contains("at least one input"));
        assert!(parse(&[ident("p"), ident("a"), string("{oops")]).unwrap_err().contains("invalid JSON"));
        assert!(parse(&[ident("p"), string("{}"), ident("a")]).unwrap_err().contains("last argument"));
        assert!(parse(&[ident("p"), ModArg::Int(3)]).unwrap_err().contains("unexpected number"));
        assert!(parse(&[ModArg::Int(1), ident("a")]).unwrap_err().contains("processor"));
        assert!(parse(&[ident("p"), ident("a"), ident("a")]).unwrap_err().contains("more than once"));
    }

    #[test]
    fn json_array_is_not_an_object() {
        let d = parse(&[ident("p"), string("[1]")]).unwrap();
        assert_eq!(d.inputs, vec!["[1]"]);
        assert!(parse(&[ident("p"), ident("a"), string(" {\"x\": 1} ")]).is_ok());
    }
}
