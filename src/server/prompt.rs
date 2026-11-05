use super::tools::Tool;
use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;

pub struct Prompt {
    environment: Environment<'static>,
}
impl Prompt {
    pub fn new(template: String) -> anyhow::Result<Self> {
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment.add_template_owned("chat", template)?;
        Ok(Self { environment })
    }
    pub fn render(
        &self,
        messages: &[Value],
        tools: &[Tool],
        thinking: bool,
    ) -> Result<String, String> {
        self.environment
            .get_template("chat")
            .map_err(|e| e.to_string())?
            .render(context! {
                messages => messages, tools => tools, enable_thinking => thinking,
                add_generation_prompt => true, add_vision_id => false,
                bos_token => "<|endoftext|>", eos_token => "<|im_end|>",
            })
            .map_err(|e| format!("chat template: {e:#}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn upstream_templates_render_text_thinking_and_tool_replay() {
        let tools:Vec<Tool>=serde_json::from_value(json!([{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}])).unwrap();
        for source in [
            include_str!("../../tests/fixtures/qwen3_chat.jinja"),
            include_str!("../../tests/fixtures/qwen35_chat.jinja"),
        ] {
            let p = Prompt::new(source.into()).unwrap();
            let messages = [
                json!({"role":"system","content":"system instruction"}),
                json!({"role":"user","content":"Read the file"}),
            ];
            let plain = p.render(&messages, &[], false).unwrap();
            assert!(plain.contains("<|im_start|>system\nsystem instruction"));
            assert!(plain.ends_with("<think>\n\n</think>\n\n"));
            let thinking = p.render(&messages, &[], true).unwrap();
            assert!(!thinking.ends_with("<think>\n\n</think>\n\n"));
            let mut messages = messages.to_vec();
            messages.push(json!({"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"read","arguments":{"path":"a.rs"}}}]}));
            messages.push(json!({"role":"tool","content":"file contents"}));
            let rendered = p.render(&messages, &tools, false).unwrap();
            assert!(rendered.contains("<tool_call>"));
            assert!(rendered.contains("<tool_response>\nfile contents\n</tool_response>"));
            assert!(rendered.contains("a.rs"));
            assert!(rendered.contains("<tools>"));
        }
    }
}
