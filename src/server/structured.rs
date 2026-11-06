//! Prompt guidance plus validation for complete structured Responses answers.
//! This does not change the model's greedy decoder or constrain token logits.
use jsonschema::{Retrieve, Uri, Validator};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct StructuredOutput {
    validator: Arc<Validator>,
    schema: Value,
}
impl StructuredOutput {
    pub fn new(format: &Value) -> Result<Self, String> {
        let object = format.as_object().ok_or("text.format must be an object")?;
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "type" | "name" | "schema" | "strict" | "description"
            ) {
                return Err(format!("unsupported JSON schema format option: {key}"));
            }
        }
        let name = format["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("json_schema format requires a name")?;
        if name.len() > 64
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err(
                "JSON schema name must contain 1–64 letters, digits, underscores or hyphens".into(),
            );
        }
        if format
            .get("strict")
            .is_some_and(|v| !v.is_null() && !v.is_boolean())
        {
            return Err("text.format.strict must be boolean".into());
        }
        if format
            .get("description")
            .is_some_and(|v| !v.is_null() && !v.is_string())
        {
            return Err("text.format.description must be a string".into());
        }
        let schema = format
            .get("schema")
            .filter(|v| v.is_object())
            .ok_or("json_schema format requires an object schema")?;
        let validator = jsonschema::options()
            .with_retriever(NoExternalReferences)
            .build(schema)
            .map_err(|e| format!("invalid output JSON schema: {e}"))?;
        Ok(Self {
            validator: Arc::new(validator),
            schema: schema.clone(),
        })
    }
    pub fn instruction(&self) -> String {
        let mut limits = String::new();
        short_string_guidance(&self.schema, "$", &mut limits);
        format!(
            "When returning an answer rather than calling a tool, return ONLY a JSON value conforming to this JSON Schema. Do not include markdown fences or surrounding explanation. Satisfy all constraints, including required properties, additionalProperties and string length limits.\n{limits}\nJSON Schema:\n{}",
            self.schema
        )
    }
    pub fn correction_instruction(&self) -> String {
        format!(
            "The prior candidate failed validation. Generate a different, more concise JSON answer. Use the shortest strings compatible with the schema. Do not copy a previous candidate.\n{}",
            self.instruction()
        )
    }
    pub fn validate(&self, text: &str) -> Result<(), String> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| format!("model output is not valid JSON for the requested schema: {e}"))?;
        self.validator.validate(&value).map_err(|e| {
            format!(
                "model output does not match the requested JSON schema at {}: {e}",
                e.instance_path
            )
        })
    }
}
fn short_string_guidance(schema: &Value, path: &str, out: &mut String) {
    if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
        let min = schema.get("minLength").and_then(Value::as_u64).unwrap_or(0);
        let target = min.max(max / 2).min(max);
        out.push_str(&format!("For {path}, the hard maximum is {max} characters INCLUDING spaces. Prefer at most {target} characters if compatible with the other constraints; use short words.\n"));
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property) in properties {
            short_string_guidance(property, &format!("{path}.{name}"), out);
        }
    }
}
struct NoExternalReferences;
impl Retrieve for NoExternalReferences {
    fn retrieve(
        &self,
        _uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "output schemas must be self-contained; external references are unsupported",
        )
        .into())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;
    pub fn title_format() -> Value {
        json!({"type":"json_schema","strict":true,"schema":{"type":"object","properties":{"title":{"type":"string","minLength":1,"maxLength":36}},"required":["title"],"additionalProperties":false},"name":"codex_output_schema"})
    }
    #[test]
    fn exact_codex_schema_rejects_invalid_titles_and_extra_properties() {
        let schema = StructuredOutput::new(&title_format()).unwrap();
        for value in [
            json!({"title":"Fix server compatibility"}),
            json!({"title":"é".repeat(36)}),
        ] {
            assert!(schema.validate(&value.to_string()).is_ok());
        }
        for value in [
            json!({}),
            json!({"title":""}),
            json!({"title":"x".repeat(37)}),
            json!({"title":7}),
            json!({"title":"ok","extra":1}),
        ] {
            assert!(schema.validate(&value.to_string()).is_err(), "{value}");
        }
        assert!(schema.validate("```json\n{\"title\":\"ok\"}\n```").is_err());
    }
    #[test]
    fn invalid_or_external_schemas_fail_before_inference() {
        for schema in [
            json!({"type":"invalid"}),
            json!({"$ref":"https://example.com/schema"}),
            json!({"$ref":"file:///tmp/schema"}),
        ] {
            let mut format = title_format();
            format["schema"] = schema;
            assert!(StructuredOutput::new(&format).is_err());
        }
        let mut format = title_format();
        format["schema"] = json!({"$defs":{"value":{"type":"string"}},"type":"object","properties":{"title":{"$ref":"#/$defs/value"}}});
        assert!(
            StructuredOutput::new(&format)
                .unwrap()
                .validate("{\"title\":\"ok\"}")
                .is_ok()
        );
    }
}
