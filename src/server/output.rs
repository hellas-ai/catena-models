//! Incremental filtering independent of the HTTP transport and GPU backend.
use super::{
    engine::Delta,
    tools::{self, Tool},
};

pub struct StopFilter {
    stops: Vec<String>,
    pending: String,
    pub stopped: bool,
}
impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            stopped: false,
        }
    }
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some(index) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            self.stopped = true;
            let text = self.pending[..index].to_owned();
            self.pending.clear();
            return text;
        }
        let hold = prefix_suffix(&self.pending, self.stops.iter().map(String::as_str));
        let end = self.pending.len() - hold;
        self.pending.drain(..end).collect()
    }
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn prefix_suffix<'a>(text: &str, markers: impl Iterator<Item = &'a str>) -> usize {
    markers
        .flat_map(|m| m.char_indices().skip(1).map(move |(n, _)| &m[..n]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

fn tool_end(text: &str) -> Option<usize> {
    if text.trim_start().starts_with("<function=") {
        return text
            .match_indices("</tool_call>")
            .find(|(i, _)| text[..*i].trim_end().ends_with("</function>"))
            .map(|(i, _)| i);
    }
    // Closing tags inside JSON string arguments are ordinary tool data.
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
        } else if ch == '"' {
            quoted = true;
        } else if text[index..].starts_with("</tool_call>") {
            return Some(index);
        }
    }
    None
}

#[derive(PartialEq)]
enum Mode {
    Text,
    Reasoning,
    Tool,
}
pub struct OutputParser {
    mode: Mode,
    pending: String,
    tools: Vec<Tool>,
    parallel: bool,
    pub calls: usize,
}
impl OutputParser {
    pub fn new(thinking: bool, tools: Vec<Tool>, parallel: bool) -> Self {
        Self {
            mode: if thinking {
                Mode::Reasoning
            } else {
                Mode::Text
            },
            pending: String::new(),
            tools,
            parallel,
            calls: 0,
        }
    }
    pub fn push(&mut self, text: &str) -> Result<Vec<Delta>, String> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.mode == Mode::Tool {
                let Some(end) = tool_end(&self.pending) else {
                    break;
                };
                let call = tools::parse_call(&self.pending[..end], &self.tools)?;
                if self.calls > 0 && !self.parallel {
                    return Err(
                        "model emitted multiple calls with parallel_tool_calls=false".into(),
                    );
                }
                out.push(Delta::Tool {
                    index: self.calls,
                    call,
                });
                self.calls += 1;
                self.pending.drain(..end + "</tool_call>".len());
                self.mode = Mode::Text;
                continue;
            }
            let markers: &[&str] = if self.mode == Mode::Reasoning {
                &["</think>", "<think>"]
            } else {
                &["<tool_call>", "<think>"]
            };
            if let Some((index, marker)) = markers
                .iter()
                .filter_map(|&m| self.pending.find(m).map(|i| (i, m)))
                .min_by_key(|(i, _)| *i)
            {
                let content: String = self.pending.drain(..index).collect();
                self.emit(content, &mut out);
                self.pending.drain(..marker.len());
                self.mode = match marker {
                    "<tool_call>" => Mode::Tool,
                    "<think>" => Mode::Reasoning,
                    _ => Mode::Text,
                };
            } else {
                let end =
                    self.pending.len() - prefix_suffix(&self.pending, markers.iter().copied());
                let content = self.pending.drain(..end).collect();
                self.emit(content, &mut out);
                break;
            }
        }
        Ok(out)
    }
    fn emit(&self, text: String, out: &mut Vec<Delta>) {
        if !text.is_empty() {
            out.push(if self.mode == Mode::Reasoning {
                Delta::Reasoning(text)
            } else {
                Delta::Content(text)
            });
        }
    }
    pub fn finish(&mut self, length: bool) -> Result<Vec<Delta>, String> {
        if self.mode == Mode::Tool {
            if length {
                self.pending.clear();
                return Ok(vec![]);
            }
            return Err("model output ended inside a tool call".into());
        }
        let text = std::mem::take(&mut self.pending);
        let mut out = Vec::new();
        self.emit(text, &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn stops_across_unicode_chunks() {
        let mut f = StopFilter::new(vec!["世界!".into(), "STOP".into()]);
        assert_eq!(f.push("hello 世"), "hello ");
        assert_eq!(f.push("界"), "");
        assert_eq!(f.push("!discard"), "");
        assert!(f.stopped);
        assert_eq!(f.finish(), "");
        let mut f = StopFilter::new(vec!["STOP".into()]);
        assert_eq!(f.push("ST"), "");
        assert_eq!(f.finish(), "ST");
    }
    #[test]
    fn split_reasoning_and_tool_delimiters() {
        let tools = serde_json::from_value(
            json!([{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}]),
        )
        .unwrap();
        let mut p = OutputParser::new(true, tools, true);
        let mut events = Vec::new();
        for ch in "reason</think>answer<tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>"
            .chars()
        {
            events.extend(p.push(&ch.to_string()).unwrap());
        }
        events.extend(p.finish(false).unwrap());
        let content: String = events
            .iter()
            .filter_map(|d| {
                if let Delta::Content(t) = d {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        let reasoning: String = events
            .iter()
            .filter_map(|d| {
                if let Delta::Reasoning(t) = d {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(content, "answer");
        assert_eq!(reasoning, "reason");
        assert_eq!(p.calls, 1);
        assert!(matches!(events.last(), Some(Delta::Tool { index: 0, .. })));
        assert!(p.push("<tool_call>broken").is_ok());
        assert!(p.finish(false).is_err());
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn opening_think_and_closing_tag_inside_json_argument() {
        let tools = serde_json::from_value(
            json!([{"type":"function","function":{"name":"write","parameters":{"type":"object"}}}]),
        )
        .unwrap();
        let mut parser = OutputParser::new(true, tools, true);
        let mut events = Vec::new();
        let text = "<think>reason</think><tool_call>{\"name\":\"write\",\"arguments\":{\"text\":\"literal </tool_call>\"}}</tool_call>";
        for ch in text.chars() {
            events.extend(parser.push(&ch.to_string()).unwrap());
        }
        events.extend(parser.finish(false).unwrap());
        let reason: String = events
            .iter()
            .filter_map(|d| {
                if let Delta::Reasoning(s) = d {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(reason, "reason");
        assert_eq!(parser.calls, 1);
        let Delta::Tool { call, .. } = events.last().unwrap() else {
            panic!("no call")
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap()["text"],
            "literal </tool_call>"
        );
    }
    #[test]
    fn truncated_tool_is_not_returned_and_parallel_limit_is_enforced() {
        let tools: Vec<Tool> =
            serde_json::from_value(json!([{"type":"function","function":{"name":"x"}}])).unwrap();
        let mut p = OutputParser::new(false, tools.clone(), false);
        p.push("<tool_call>incomplete").unwrap();
        assert!(p.finish(true).unwrap().is_empty());
        let mut p = OutputParser::new(false, tools, false);
        let call = "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>";
        p.push(call).unwrap();
        assert!(p.push(call).is_err());
    }
}
