//! Model tool syntax ↔ normalized function calls. Tools always execute on clients.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Function,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "empty_schema")]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
fn empty_schema() -> Value {
    json!({"type":"object", "properties":{}})
}

pub fn validate_tools(tools: &[Tool]) -> Result<(), String> {
    let mut names = BTreeSet::new();
    for tool in tools {
        let f = &tool.function;
        if tool.kind != "function"
            || f.name.is_empty()
            || f.name.len() > 64
            || !f
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err(
                "tools must be named function tools (1–64 letters, digits, '_' or '-')".into(),
            );
        }
        if !names.insert(&f.name) {
            return Err("duplicate tool name".into());
        }
        if f.strict == Some(true) {
            return Err("strict constrained tool output is not supported".into());
        }
        if !f.parameters.is_object() || f.parameters.get("type").is_some_and(|x| x != "object") {
            return Err("tool parameters must be an object schema".into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
pub struct Call {
    pub name: String,
    pub arguments: String,
}

pub fn parse_call(body: &str, tools: &[Tool]) -> Result<Call, String> {
    let body = body.trim();
    let (name, arguments) = if body.starts_with("<function=") {
        parse_xml(body, tools)?
    } else {
        let v: Value = serde_json::from_str(body).map_err(|e| format!("invalid tool JSON: {e}"))?;
        let name = v["name"]
            .as_str()
            .ok_or("tool call has no name")?
            .to_owned();
        let arguments = v
            .get("arguments")
            .cloned()
            .ok_or("tool call has no arguments")?;
        let arguments = if let Some(s) = arguments.as_str() {
            serde_json::from_str(s).map_err(|e| format!("invalid tool arguments: {e}"))?
        } else {
            arguments
        };
        (name, arguments)
    };
    let tool = tools
        .iter()
        .find(|t| t.function.name == name)
        .ok_or("model called an undeclared tool")?;
    let args = arguments
        .as_object()
        .ok_or("tool arguments must be a JSON object")?;
    validate_arguments(args, &tool.function.parameters)?;
    Ok(Call {
        name,
        arguments: arguments.to_string(),
    })
}

fn parse_xml(body: &str, tools: &[Tool]) -> Result<(String, Value), String> {
    let (name, tail) = body
        .strip_prefix("<function=")
        .ok_or("missing function tag")?
        .split_once('>')
        .ok_or("incomplete function tag")?;
    let tool = tools
        .iter()
        .find(|t| t.function.name == name)
        .ok_or("model called an undeclared tool")?;
    let mut rest = tail.trim_start();
    let mut args = Map::new();
    while let Some(after) = rest.strip_prefix("<parameter=") {
        let (key, after) = after.split_once('>').ok_or("incomplete parameter tag")?;
        let (value, after) = after
            .split_once("</parameter>")
            .ok_or("unclosed parameter tag")?;
        if args.contains_key(key) {
            return Err("duplicate tool parameter".into());
        }
        // Remove the template's delimiter newlines, preserving string whitespace.
        let raw = value.strip_prefix('\n').unwrap_or(value);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let schema = &tool.function.parameters["properties"][key];
        let value = xml_value(raw, schema)?;
        args.insert(key.to_owned(), value);
        rest = after.trim_start();
    }
    if rest.trim() != "</function>" {
        return Err("malformed function output".into());
    }
    Ok((name.to_owned(), Value::Object(args)))
}

fn xml_value(raw: &str, schema: &Value) -> Result<Value, String> {
    if schema["type"] == "string" {
        return Ok(Value::String(raw.into()));
    }
    if let Some(variants) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        for variant in variants {
            if let Ok(v) = xml_value(raw, variant)
                && matches_type(&v, variant)
            {
                return Ok(v);
            }
        }
        return Err("tool parameter does not match declared alternatives".into());
    }
    let parsed = serde_json::from_str(raw.trim());
    if schema.get("type").is_none() {
        return Ok(parsed.unwrap_or_else(|_| Value::String(raw.into())));
    }
    let value = parsed.map_err(|e| format!("invalid typed tool parameter: {e}"))?;
    if !matches_type(&value, schema) {
        return Err("tool parameter type mismatch".into());
    }
    Ok(value)
}
fn matches_type(value: &Value, schema: &Value) -> bool {
    let matches = |ty: &str| match ty {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    };
    if let Some(ty) = schema["type"].as_str() {
        matches(ty)
    } else if let Some(types) = schema["type"].as_array() {
        types.iter().any(|t| t.as_str().is_some_and(matches))
    } else {
        true
    }
}
fn validate_arguments(args: &Map<String, Value>, schema: &Value) -> Result<(), String> {
    if let Some(required) = schema["required"].as_array() {
        for key in required {
            if !key.as_str().is_some_and(|key| args.contains_key(key)) {
                return Err("model omitted a required tool argument".into());
            }
        }
    }
    for (key, value) in args {
        let property = &schema["properties"][key];
        if property.is_null() && schema["additionalProperties"] == false {
            return Err("model emitted an unknown tool argument".into());
        }
        if !matches_type(value, property) {
            return Err("tool argument type mismatch".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tools() -> Vec<Tool> {
        serde_json::from_value(json!([{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"},"flag":{"type":"boolean"},"data":{"type":"object"}},"required":["text"]}}}])).unwrap()
    }
    #[test]
    fn both_formats_preserve_types() {
        let t = tools();
        let call=parse_call("<function=edit>\n<parameter=text>\n123\n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=data>\n{\"x\":1}\n</parameter>\n</function>",&t).unwrap();
        let args: Value = serde_json::from_str(&call.arguments).unwrap();
        assert_eq!(
            args,
            json!({"text":"123","count":2,"flag":true,"data":{"x":1}})
        );
        assert!(parse_call(r#"{"name":"edit","arguments":{"text":"hi"}}"#, &t).is_ok());
        assert!(parse_call(r#"{"name":"edit","arguments":"{\"text\":\"hi\"}"}"#, &t).is_ok());
        for body in [
            r#"{"name":"bad","arguments":{}}"#,
            r#"{"name":"edit","arguments":{}}"#,
            "<function=edit><parameter=count>bad</parameter></function>",
            "<function=edit>",
            "{}",
        ] {
            assert!(parse_call(body, &t).is_err(), "{body}");
        }
    }
}
