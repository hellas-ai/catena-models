use super::{
    api::{ServerState, router},
    engine::{self, Backend, Delta, Event, Failure, Generation, Job, Sink},
    responses::Request,
    tools::Call,
};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request as HttpRequest, StatusCode},
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tower::ServiceExt;

fn base() -> Value {
    json!({"model":"test","input":"hello","store":false})
}
fn request(mut extra: Value) -> Request {
    let mut value = base();
    value
        .as_object_mut()
        .unwrap()
        .append(extra.as_object_mut().unwrap());
    serde_json::from_value(value).unwrap()
}
fn function() -> Value {
    json!({"type":"function","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}})
}
fn custom() -> Value {
    json!({"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}})
}
fn post(value: Value) -> HttpRequest<Body> {
    HttpRequest::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-key")
        .body(Body::from(value.to_string()))
        .unwrap()
}
async fn text(response: axum::response::Response) -> String {
    String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}
async fn json_body(response: axum::response::Response) -> Value {
    serde_json::from_str(&text(response).await).unwrap()
}
fn events(wire: &str) -> Vec<Value> {
    let events: Vec<Value> = wire
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .map(|frame| {
            let mut name = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(s) = line.strip_prefix("event: ") {
                    name = Some(s);
                }
                if let Some(s) = line.strip_prefix("data: ") {
                    data = Some(s);
                }
            }
            let value: Value = serde_json::from_str(data.unwrap()).unwrap();
            assert_eq!(Some(value["type"].as_str().unwrap()), name);
            value
        })
        .collect();
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], index);
    }
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[1]["type"], "response.in_progress");
    events
}

#[test]
fn codex_client_metadata_is_accepted_without_changing_model_input() {
    let original = request(json!({})).prepare("test").unwrap();
    for metadata in [
        json!({"session_id":"session-1","turn_id":"turn-2"}),
        Value::Null,
    ] {
        let prepared = request(json!({"client_metadata":metadata}))
            .prepare("test")
            .unwrap();
        assert_eq!(prepared.generation.messages, original.generation.messages);
        assert_eq!(prepared.settings, original.settings);
    }
    assert!(
        request(json!({"client_metadata":"invalid"}))
            .prepare("test")
            .is_err()
    );
}

#[test]
fn nullable_codex_text_controls_use_plain_text_defaults() {
    let original = request(json!({})).prepare("test").unwrap();
    for text in [
        json!({"format":null}),
        json!({"format":null,"verbosity":null}),
        json!({"format":{"type":"text","name":null,"schema":null,"strict":null}}),
    ] {
        let prepared = request(json!({"text":text})).prepare("test").unwrap();
        assert_eq!(prepared.generation.messages, original.generation.messages);
        assert_eq!(prepared.settings, original.settings);
    }
    let prepared = request(json!({"text":{"format":null,"verbosity":"low"}}))
        .prepare("test")
        .unwrap();
    assert_eq!(prepared.settings["text"]["verbosity"], "low");
    for format in [
        json!({"type":"json_object"}),
        json!({"type":"text","schema":{"type":"object"}}),
    ] {
        assert!(
            request(json!({"text":{"format":format}}))
                .prepare("test")
                .is_err()
        );
    }
}

#[test]
fn namespace_groups_are_ignored_without_exposing_nested_tools() {
    let prepared = request(json!({"tools":[
        {"type":"namespace","name":"mcp__codex_worktrees","description":"Worktrees","tools":[
            {"type":"function","name":"create_worktree","parameters":{"type":"object"}},
            {"type":"function","name":"read_file","parameters":{"type":"object"}}
        ]},
        function(), custom(), {"type":"namespace"}
    ]}))
    .prepare("test")
    .unwrap();
    let names: Vec<_> = prepared
        .generation
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    assert_eq!(names, ["read_file", "apply_patch"]);
    assert_eq!(prepared.settings["tools"], json!([function(), custom()]));
    assert!(
        request(json!({"tools":[{"type":"namespace","tools":[]}]}))
            .prepare("test")
            .unwrap()
            .generation
            .tools
            .is_empty()
    );
}

#[test]
fn translates_messages_instructions_tools_and_reasoning() {
    let prepared=request(json!({"instructions":"system","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[function(),custom()],"reasoning":{"effort":"medium","summary":"auto"},"include":["reasoning.encrypted_content"],"text":{"format":{"type":"text"},"verbosity":"low"}})).prepare("test").unwrap();
    assert!(prepared.generation.thinking);
    assert_eq!(prepared.generation.tools.len(), 2);
    assert!(prepared.custom.contains("apply_patch"));
    assert!(
        prepared.generation.messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("system")
    );
    assert_eq!(prepared.generation.messages[1]["content"], "hi");
    assert_eq!(
        prepared.generation.tools[1].function.parameters["properties"]["input"]["type"],
        "string"
    );
    assert!(
        prepared.generation.tools[1]
            .function
            .description
            .as_ref()
            .unwrap()
            .contains("start:")
    );
    for extra in [
        json!({"store":true}),
        json!({"previous_response_id":"resp_old"}),
        json!({"background":true}),
        json!({"truncation":"auto"}),
        json!({"input":[{"type":"item_reference","id":"msg_old"}]}),
        json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}),
        json!({"tools":[{"type":"web_search"}]}),
        json!({"reasoning":{"effort":"impossible"}}),
        json!({"temperature":1}),
        json!({"input":[{"type":"reasoning","encrypted_content":"opaque"}]}),
        json!({"text":{"format":{"type":"json_object"}}}),
    ] {
        assert!(request(extra.clone()).prepare("test").is_err(), "{extra}");
    }
}
#[test]
fn replay_multiple_calls_and_custom_results_preserves_ids() {
    let prepared=request(json!({"input":[{"role":"user","content":"read"},
        {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"plan"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"checking"}]},
        {"type":"function_call","id":"fc_1","call_id":"a","name":"read_file","arguments":"{\"path\":\"a.rs\"}"},
        {"type":"custom_tool_call","id":"ctc_1","call_id":"b","name":"apply_patch","input":"*** Begin Patch\n*** End Patch"},
        {"type":"function_call_output","call_id":"a","output":[{"type":"input_text","text":"contents"}]},
        {"type":"custom_tool_call_output","call_id":"b","output":"success"}
    ]})).prepare("test").unwrap();
    let messages = prepared.generation.messages;
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[1]["reasoning_content"], "plan");
    assert_eq!(messages[1]["content"], "checking");
    assert_eq!(messages[1]["tool_calls"][0]["id"], "a");
    assert_eq!(messages[1]["tool_calls"][1]["id"], "b");
    assert_eq!(
        messages[1]["tool_calls"][1]["function"]["arguments"]["input"],
        "*** Begin Patch\n*** End Patch"
    );
    assert_eq!(messages[3]["tool_call_id"], "b");
    assert!(request(json!({"input":[{"role":"user","content":"hi"},{"type":"function_call_output","call_id":"missing","output":"x"}]})).prepare("test").is_err());
}

struct Fake {
    cancelled: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}
impl Backend for Fake {
    fn generate(&mut self, request: Generation, sink: &Sink) -> Result<(), Failure> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if request.max_tokens == Some(999) {
            return Err(Failure::invalid("context overflow"));
        }
        sink.send(Event::Started);
        if request.max_tokens == Some(998) {
            return Err(Failure::internal("GPU failure"));
        }
        if request.max_tokens == Some(997) {
            while sink.send(Event::Delta(Delta::Content("slow".into()))) {}
            self.cancelled.store(true, Ordering::Relaxed);
            return Ok(());
        }
        if request.thinking {
            sink.send(Event::Delta(Delta::Reasoning("plan".into())));
        }
        sink.send(Event::Delta(Delta::Content("Hello 世界".into())));
        for (index, tool) in request.tools.iter().enumerate() {
            let arguments = if tool.function.parameters["properties"]["input"]["type"] == "string" {
                json!({"input":"*** Begin Patch\n*** End Patch"})
            } else {
                json!({"path":"a.rs"})
            };
            sink.send(Event::Delta(Delta::Tool {
                index,
                call: Call {
                    name: tool.function.name.clone(),
                    arguments: arguments.to_string(),
                },
            }));
        }
        sink.send(Event::Done {
            reason: if request.max_tokens == Some(1) {
                "length"
            } else {
                "stop"
            },
            prompt_tokens: 5,
            completion_tokens: 4,
            reasoning_tokens: usize::from(request.thinking),
            cached_tokens: 3,
            cache_write_tokens: 2,
        });
        Ok(())
    }
}
fn app() -> (
    Router,
    std::thread::JoinHandle<()>,
    Arc<AtomicBool>,
    Arc<AtomicUsize>,
) {
    let cancelled = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let c = cancelled.clone();
    let n = calls.clone();
    let worker = engine::start(
        move || {
            Ok(Fake {
                cancelled: c,
                calls: n,
            })
        },
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    (
        router(ServerState {
            model: Arc::from("test"),
            key: Some(Arc::from("test-key")),
            jobs: worker.sender,
        }),
        worker.thread,
        cancelled,
        calls,
    )
}

#[tokio::test]
async fn json_and_sse_text_tools_reasoning_and_usage() {
    let (app, worker, _, _) = app();
    let mut request = base();
    request["client_metadata"] = json!({"session_id":"codex-session","turn_id":"codex-turn"});
    request["text"] = json!({"format":null,"verbosity":"low"});
    request["tools"] = json!([function(), custom()]);
    request["reasoning"] = json!({"effort":"medium"});
    let response = app.clone().oneshot(post(request.clone())).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["store"], false);
    assert_eq!(body["output"][0]["type"], "reasoning");
    assert_eq!(body["output"][0]["summary"][0]["text"], "plan");
    assert_eq!(body["output"][1]["content"][0]["text"], "Hello 世界");
    assert_eq!(body["output"][2]["type"], "function_call");
    assert_eq!(body["output"][2]["arguments"], "{\"path\":\"a.rs\"}");
    assert_eq!(body["output"][3]["type"], "custom_tool_call");
    assert_eq!(body["output"][3]["input"], "*** Begin Patch\n*** End Patch");
    assert_ne!(body["output"][2]["id"], body["output"][2]["call_id"]);
    assert_eq!(body["usage"]["input_tokens"], 5);
    assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 3);
    assert_eq!(
        body["usage"]["input_tokens_details"]["cache_write_tokens"],
        2
    );
    assert_eq!(
        body["usage"]["output_tokens_details"]["reasoning_tokens"],
        1
    );
    // A response's output array is valid replay input when the caller supplies matching results.
    let mut input = vec![json!({"role":"user","content":"hello"})];
    input.extend(body["output"].as_array().unwrap().clone());
    input.push(json!({"type":"function_call_output","call_id":body["output"][2]["call_id"],"output":"contents"}));
    input.push(json!({"type":"custom_tool_call_output","call_id":body["output"][3]["call_id"],"output":"applied"}));
    assert!(
        super::responses_tests::request(json!({"input":input}))
            .prepare("test")
            .is_ok()
    );
    request["stream"] = json!(true);
    let response = app.clone().oneshot(post(request)).await.unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let raw = text(response).await;
    assert!(!raw.contains("[DONE]"));
    let stream = events(&raw);
    let last = stream.last().unwrap();
    assert_eq!(last["type"], "response.completed");
    assert_eq!(last["response"]["usage"], body["usage"]);
    let output = &last["response"]["output"];
    for index in 0..4 {
        assert_eq!(output[index]["type"], body["output"][index]["type"]);
    }
    let args = stream
        .iter()
        .find(|e| e["type"] == "response.function_call_arguments.delta")
        .unwrap();
    assert_eq!(args["output_index"], 2);
    assert_eq!(args["item_id"], output[2]["id"]);
    assert_eq!(
        stream
            .iter()
            .filter(|e| e["type"] == "response.output_item.added")
            .count(),
        4
    );
    assert_eq!(
        stream
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .count(),
        4
    );
    assert_eq!(
        stream
            .iter()
            .filter(|e| e["type"] == "response.custom_tool_call_input.delta")
            .count(),
        1
    );
    assert!(
        stream
            .iter()
            .any(|e| e["type"] == "response.reasoning_summary_text.delta")
    );
    assert!(
        stream
            .iter()
            .any(|e| e["type"] == "response.content_part.done")
    );
    drop(app);
    worker.join().unwrap();
}
#[tokio::test]
async fn failures_limits_and_cancellation_do_not_emit_success() {
    let (app, worker, cancelled, _) = app();
    let mut req = base();
    req["max_output_tokens"] = json!(999);
    assert_eq!(
        app.clone().oneshot(post(req)).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    let mut req = base();
    req["max_output_tokens"] = json!(998);
    assert_eq!(
        app.clone()
            .oneshot(post(req.clone()))
            .await
            .unwrap()
            .status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    req["stream"] = json!(true);
    let wire = events(&text(app.clone().oneshot(post(req)).await.unwrap()).await);
    assert_eq!(wire.last().unwrap()["type"], "response.failed");
    assert_eq!(
        wire.last().unwrap()["response"]["error"]["message"],
        "GPU failure"
    );
    assert!(!wire.iter().any(|e| e["type"] == "response.completed"));
    let mut req = base();
    req["max_output_tokens"] = json!(1);
    req["stream"] = json!(true);
    let wire = events(&text(app.clone().oneshot(post(req)).await.unwrap()).await);
    assert_eq!(wire.last().unwrap()["type"], "response.incomplete");
    assert_eq!(
        wire.last().unwrap()["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(
        wire.last().unwrap()["response"]["output"][0]["status"],
        "incomplete"
    );
    let mut req = base();
    req["max_output_tokens"] = json!(997);
    req["stream"] = json!(true);
    drop(app.clone().oneshot(post(req)).await.unwrap());
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !cancelled.load(Ordering::Relaxed) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        app.clone().oneshot(post(base())).await.unwrap().status(),
        StatusCode::OK
    );
    drop(app);
    worker.join().unwrap();
}
#[tokio::test]
async fn responses_share_authentication_body_limits_and_queue() {
    let (app, worker, _, _) = app();
    let mut unauth = post(base());
    unauth.headers_mut().remove("authorization");
    assert_eq!(
        app.clone().oneshot(unauth).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let mut req = base();
    req["model"] = json!("wrong");
    assert_eq!(
        app.clone().oneshot(post(req)).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
    let mut req = base();
    req["store"] = json!(true);
    assert_eq!(
        app.clone().oneshot(post(req)).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    drop(app);
    worker.join().unwrap();
    let (tx, rx) = tokio::sync::mpsc::channel::<Job>(1);
    let (out, _) = tokio::sync::mpsc::channel(1);
    tx.try_send(Job {
        request: request(json!({})).prepare("test").unwrap().generation,
        output: out,
    })
    .unwrap();
    let app = router(ServerState {
        model: Arc::from("test"),
        key: None,
        jobs: tx,
    });
    assert_eq!(
        app.clone().oneshot(post(base())).await.unwrap().status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    drop(rx);
    assert_eq!(
        app.oneshot(post(base())).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn strict_codex_titles_are_validated_for_json_and_sse() {
    struct Titles;
    impl Backend for Titles {
        fn generate(&mut self, request: Generation, sink: &Sink) -> Result<(), Failure> {
            let mode = request.messages.last().unwrap()["content"]
                .as_str()
                .unwrap();
            assert!(
                request.messages[0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("maxLength")
            );
            sink.send(Event::Started);
            if request.thinking {
                sink.send(Event::Delta(Delta::Reasoning("choose a title".into())));
            }
            let answer = match mode {
                "valid" => json!({"title":"Fix Codex structured outputs"}).to_string(),
                "empty" => json!({"title":""}).to_string(),
                "long" => json!({"title":"x".repeat(37)}).to_string(),
                "extra" => json!({"title":"ok","extra":true}).to_string(),
                "missing" => "{}".into(),
                "malformed" => "not JSON".into(),
                "length" => "{\"title\":".into(),
                "tool" => "Checking the file first.".into(),
                _ => panic!("unknown test case"),
            };
            for ch in answer.chars() {
                if !sink.send(Event::Delta(Delta::Content(ch.to_string()))) {
                    return Ok(());
                }
            }
            if mode == "tool" {
                sink.send(Event::Delta(Delta::Tool {
                    index: 0,
                    call: Call {
                        name: "read_file".into(),
                        arguments: "{\"path\":\"a.rs\"}".into(),
                    },
                }));
            }
            sink.send(Event::Done {
                reason: if mode == "length" {
                    "length"
                } else if mode == "tool" {
                    "tool_calls"
                } else {
                    "stop"
                },
                prompt_tokens: 50,
                completion_tokens: 10,
                reasoning_tokens: usize::from(request.thinking),
                cached_tokens: 0,
                cache_write_tokens: 0,
            });
            Ok(())
        }
    }
    let worker = engine::start(|| Ok(Titles), Arc::new(AtomicBool::new(false))).unwrap();
    let app = router(ServerState {
        model: Arc::from("test"),
        key: None,
        jobs: worker.sender,
    });
    let format = super::structured::tests::title_format();
    let prepared = request(json!({"text":{"format":format}}))
        .prepare("test")
        .unwrap();
    assert!(prepared.structured.is_some());
    assert_eq!(prepared.settings["text"]["format"], format);
    for streaming in [false, true] {
        for mode in [
            "valid",
            "empty",
            "long",
            "extra",
            "missing",
            "malformed",
            "length",
            "tool",
        ] {
            let mut req = base();
            req["input"] = json!(mode);
            req["text"] = json!({"format":format});
            req["stream"] = json!(streaming);
            req["reasoning"] = json!({"effort":"low"});
            if mode == "tool" {
                req["tools"] = json!([function()]);
            }
            let response = app.clone().oneshot(post(req)).await.unwrap();
            let valid = matches!(mode, "valid" | "length" | "tool");
            if streaming {
                assert_eq!(response.status(), StatusCode::OK);
                let wire = events(&text(response).await);
                let terminal = wire.last().unwrap();
                assert_eq!(
                    terminal["type"],
                    if mode == "length" {
                        "response.incomplete"
                    } else if valid {
                        "response.completed"
                    } else {
                        "response.failed"
                    }
                );
                if mode == "valid" {
                    let deltas: Vec<_> = wire
                        .iter()
                        .filter(|e| e["type"] == "response.output_text.delta")
                        .collect();
                    assert_eq!(deltas.len(), 1);
                    assert_eq!(
                        serde_json::from_str::<Value>(deltas[0]["delta"].as_str().unwrap())
                            .unwrap()["title"],
                        "Fix Codex structured outputs"
                    );
                } else if !valid {
                    assert!(
                        !wire
                            .iter()
                            .any(|e| e["type"] == "response.output_text.delta")
                    );
                    assert!(!wire.iter().any(|e| e["type"] == "response.completed"));
                }
                if mode == "tool" {
                    assert_eq!(terminal["response"]["output"][1]["type"], "message");
                    assert_eq!(terminal["response"]["output"][2]["type"], "function_call");
                }
            } else if valid {
                assert_eq!(response.status(), StatusCode::OK);
                let result = json_body(response).await;
                assert_eq!(
                    result["status"],
                    if mode == "length" {
                        "incomplete"
                    } else {
                        "completed"
                    }
                );
                if mode == "valid" {
                    assert_eq!(
                        serde_json::from_str::<Value>(
                            result["output"][1]["content"][0]["text"].as_str().unwrap()
                        )
                        .unwrap()["title"],
                        "Fix Codex structured outputs"
                    );
                }
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert!(
                    json_body(response).await["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("model output")
                );
            }
        }
    }
    drop(app);
    worker.thread.join().unwrap();
}

#[tokio::test]
async fn rejected_tools_print_complete_incoming_json_at_warn_level() {
    // Tracing callsite interest is process-global; isolate this capture from
    // other parallel tests that exercise the same rejection without a logger.
    const CHILD: &str = "CATENA_TEST_REJECTED_REQUEST_LOG";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "server::responses_tests::rejected_tools_print_complete_incoming_json_at_warn_level", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use tracing::instrument::WithSubscriber;
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let writer = super::tests::LogWriter(captured.clone());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_env_filter("catena_models::server::protocol=warn")
        .with_writer(move || writer.clone())
        .finish();
    let (app, worker, _, calls) = app();
    let mut incoming = base();
    incoming["client_metadata"] = json!({"session_id":"copy-this-session"});
    incoming["tools"] =
        json!([function(),{"type":"unsupported","name":"unsupported_tool","options":{"sample":1}}]);
    async {
        let response = app.clone().oneshot(post(incoming.clone())).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(response).await["error"]["message"],
            "Responses supports function and custom client-executed tools only"
        );
    }
    .with_subscriber(subscriber)
    .await;
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    drop(app);
    worker.join().unwrap();
    let logs = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
    assert!(
        logs.contains("Rejected Responses request"),
        "captured logs: {logs:?}"
    );
    let logged = logs
        .lines()
        .find_map(|line| line.split_once("request_json=").map(|(_, json)| json))
        .unwrap();
    assert_eq!(serde_json::from_str::<Value>(logged).unwrap(), incoming);
    assert!(!logs.contains("Bearer"));
    assert!(!logs.contains("test-key"));
}
