use std::io;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tokio::sync::Notify;

use crate::chat::{RejectedChat, accept_chat};
use crate::complete::{CompleteError, Completion, Loaded, complete};
use crate::generate::Model;

use crate::load::StartupError;

#[derive(Serialize)]
struct HealthBody {
    status: &'static str,
}

async fn health() -> Json<HealthBody> {
    Json(HealthBody { status: "ok" })
}

/// Routes that exist only after the model has loaded.
pub fn health_router() -> Router {
    serve_router("")
}

#[derive(Clone)]
struct AppState {
    model_name: String,
    engine: Option<Arc<Engine>>,
    api_key: Option<String>,
    max_body_bytes: usize,
    gate: Arc<Gate>,
    ids: Arc<AtomicU64>,
}

struct Engine {
    loaded: Loaded,
    model: Mutex<Box<dyn Model + Send>>,
}

struct Gate {
    capacity: usize,
    inner: Mutex<GateInner>,
    notify: Notify,
}

struct GateInner {
    running: bool,
    waiting: usize,
}

struct Running {
    gate: Arc<Gate>,
}

impl Drop for Running {
    fn drop(&mut self) {
        let mut gate = self.gate.inner.lock().expect("admission lock");
        gate.running = false;
        self.gate.notify.notify_one();
    }
}

impl Gate {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: Mutex::new(GateInner {
                running: false,
                waiting: 0,
            }),
            notify: Notify::new(),
        }
    }

    async fn enter(self: &Arc<Self>) -> Result<Running, ()> {
        loop {
            {
                let mut gate = self.inner.lock().expect("admission lock");
                if !gate.running {
                    gate.running = true;
                    return Ok(Running {
                        gate: Arc::clone(self),
                    });
                }
                if gate.waiting >= self.capacity {
                    return Err(());
                }
                gate.waiting += 1;
            }
            let mut waiting = Waiting {
                gate: Arc::clone(self),
                active: true,
            };
            self.notify.notified().await;
            waiting.active = false;
            let mut gate = self.inner.lock().expect("admission lock");
            gate.waiting = gate.waiting.saturating_sub(1);
        }
    }
}

struct Waiting {
    gate: Arc<Gate>,
    active: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.active {
            let mut gate = self.gate.inner.lock().expect("admission lock");
            gate.waiting = gate.waiting.saturating_sub(1);
        }
    }
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelItem>,
}

#[derive(Serialize)]
struct ModelItem {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Serialize)]
struct ApiError {
    message: &'static str,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
}

fn api_error(
    status: StatusCode,
    message: &'static str,
    error_type: &'static str,
    code: &'static str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ApiError {
                message,
                error_type,
                code,
            },
        }),
    )
}

async fn method_not_allowed() -> (StatusCode, Json<ErrorResponse>) {
    api_error(
        StatusCode::METHOD_NOT_ALLOWED,
        "method not allowed",
        "invalid_request_error",
        "method_not_allowed",
    )
}

async fn chat(State(state): State<AppState>, request: Request) -> Response {
    if let Some(key) = &state.api_key {
        let expected = format!("Bearer {key}");
        let header = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok());
        if header != Some(expected.as_str()) {
            return rejected(RejectedChat {
                status: 401,
                message: "the api key is invalid",
                code: "invalid_api_key",
            });
        }
    }
    let content_type = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !content_type.starts_with("application/json") {
        return rejected(RejectedChat {
            status: 400,
            message: "content type must be application/json",
            code: "bad_request",
        });
    }
    if request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > state.max_body_bytes)
    {
        return too_large();
    }
    let bytes = match to_bytes(request.into_body(), state.max_body_bytes).await {
        Ok(bytes) => bytes,
        Err(_) => return too_large(),
    };
    let body: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(_) => {
            return rejected(RejectedChat {
                status: 400,
                message: "the request body is invalid",
                code: "bad_request",
            });
        }
    };
    let chat = match accept_chat(&body, &state.model_name) {
        Ok(chat) => chat,
        Err(error) => return rejected(error),
    };
    let Some(engine) = state.engine.clone() else {
        return rejected(RejectedChat {
            status: 500,
            message: "the completion is not available",
            code: "internal",
        });
    };
    if let Err(error) = crate::complete::fits(&engine.loaded, &chat) {
        return completion_error(error);
    }
    let running = match state.gate.enter().await {
        Ok(running) => running,
        Err(()) => {
            return rejected(RejectedChat {
                status: 503,
                message: "the queue is full",
                code: "queue_full",
            });
        }
    };
    let stream = chat.stream;
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel_on_drop = CancelOnDrop(Arc::clone(&cancelled));
    let scored = tokio::task::spawn_blocking(move || {
        let mut model = engine.model.lock().expect("model lock");
        let result = complete(&mut **model, &engine.loaded, &chat, &cancelled);
        drop(running);
        result
    })
    .await;
    cancel_on_drop.defuse();
    let completion = match scored {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => return completion_error(error),
        Err(_) => {
            return rejected(RejectedChat {
                status: 500,
                message: "generation failed",
                code: "internal",
            });
        }
    };
    let id = format!("chatcmpl-{}", state.ids.fetch_add(1, Ordering::Relaxed));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    if stream {
        stream_response(&id, created, &state.model_name, &completion)
    } else {
        Json(completion_body(
            &id,
            created,
            &state.model_name,
            &completion,
        ))
        .into_response()
    }
}

struct CancelOnDrop(Arc<AtomicBool>);

impl CancelOnDrop {
    fn defuse(self) {
        std::mem::forget(self);
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn rejected(error: RejectedChat) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::BAD_REQUEST);
    let error_type = if error.status == 401 {
        "authentication_error"
    } else if error.status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(error_body(error.message, error_type, error.code)),
    )
        .into_response()
}

fn too_large() -> Response {
    rejected(RejectedChat {
        status: 413,
        message: "the request body is too large",
        code: "body_too_large",
    })
}

fn completion_error(error: CompleteError) -> Response {
    match error {
        CompleteError::ContextLength => rejected(RejectedChat {
            status: 400,
            message: "the prompt is longer than the working context",
            code: "context_length_exceeded",
        }),
        CompleteError::Tokenize => rejected(RejectedChat {
            status: 400,
            message: "the prompt cannot be tokenized",
            code: "bad_request",
        }),
        CompleteError::Template => rejected(RejectedChat {
            status: 500,
            message: "the chat template failed",
            code: "internal",
        }),
        CompleteError::Cancelled => rejected(RejectedChat {
            status: 500,
            message: "the request was cancelled",
            code: "internal",
        }),
        CompleteError::Generate => rejected(RejectedChat {
            status: 500,
            message: "generation failed",
            code: "internal",
        }),
    }
}

#[derive(Serialize)]
struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: UsageBody,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct UsageBody {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct Chunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageBody>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn completion_body(id: &str, created: u64, model: &str, completion: &Completion) -> ChatCompletion {
    ChatCompletion {
        id: id.to_string(),
        object: "chat.completion",
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: completion.text(),
            },
            finish_reason: completion.finish_reason,
        }],
        usage: usage_body(completion),
    }
}

fn stream_response(id: &str, created: u64, model: &str, completion: &Completion) -> Response {
    let mut body = String::new();
    for (index, piece) in completion.pieces.iter().enumerate() {
        body.push_str("data: ");
        body.push_str(
            &serde_json::to_string(&chunk(
                id,
                created,
                model,
                Delta {
                    role: (index == 0).then_some("assistant"),
                    content: Some(piece.clone()),
                },
                None,
                None,
            ))
            .expect("chunk"),
        );
        body.push_str("\n\n");
    }
    body.push_str("data: ");
    body.push_str(
        &serde_json::to_string(&chunk(
            id,
            created,
            model,
            Delta {
                role: completion.pieces.is_empty().then_some("assistant"),
                content: None,
            },
            Some(completion.finish_reason),
            Some(usage_body(completion)),
        ))
        .expect("chunk"),
    );
    body.push_str("\n\ndata: [DONE]\n\n");
    ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish_reason: Option<&'static str>,
    usage: Option<UsageBody>,
) -> Chunk {
    Chunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }
}

fn usage_body(completion: &Completion) -> UsageBody {
    UsageBody {
        prompt_tokens: completion.prompt_tokens,
        completion_tokens: completion.completion_tokens,
        total_tokens: completion
            .prompt_tokens
            .saturating_add(completion.completion_tokens),
    }
}

fn error_body(
    message: &'static str,
    error_type: &'static str,
    code: &'static str,
) -> ErrorResponse {
    ErrorResponse {
        error: ApiError {
            message,
            error_type,
            code,
        },
    }
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelItem {
            id: state.model_name,
            object: "model",
            owned_by: "local",
        }],
    })
}

/// The loaded model is the only entry `GET /v1/models` may return.
fn routes(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health).fallback(method_not_allowed))
        .route("/v1/models", get(list_models).fallback(method_not_allowed))
        .route(
            "/v1/chat/completions",
            post(chat).fallback(method_not_allowed),
        )
        .with_state(state)
}

/// Routes for a loaded name before a model is attached. A valid chat is not scored.
pub fn serve_router(model_name: &str) -> Router {
    routes(named_state(model_name, None, None, 1024 * 1024, 64))
}

/// Routes that score `model` for one client at a time.
pub fn serve_attached(
    model_name: &str,
    loaded: Loaded,
    model: Box<dyn Model + Send>,
    api_key: Option<String>,
    max_body_bytes: usize,
    queue_capacity: usize,
) -> Router {
    routes(named_state(
        model_name,
        Some(Arc::new(Engine {
            loaded,
            model: Mutex::new(model),
        })),
        api_key,
        max_body_bytes,
        queue_capacity,
    ))
}

fn named_state(
    model_name: &str,
    engine: Option<Arc<Engine>>,
    api_key: Option<String>,
    max_body_bytes: usize,
    queue_capacity: usize,
) -> AppState {
    AppState {
        model_name: model_name.to_string(),
        engine,
        api_key,
        max_body_bytes,
        gate: Arc::new(Gate::new(queue_capacity)),
        ids: Arc::new(AtomicU64::new(1)),
    }
}

/// Open `addr` on the process. An occupied port is distinct from any other bind failure.
pub fn bind_listener(addr: SocketAddr) -> Result<TcpListener, StartupError> {
    TcpListener::bind(addr).map_err(|error| classify_bind_error(addr, error))
}

pub(crate) fn classify_bind_error(addr: SocketAddr, error: io::Error) -> StartupError {
    if error.kind() == io::ErrorKind::AddrInUse {
        StartupError::PortInUse { port: addr.port() }
    } else {
        StartupError::Bind {
            address: addr.to_string(),
        }
    }
}

/// Turn a host string and a port into the address `serve` will bind.
pub fn serve_addr(host: &str, port: u16) -> Result<SocketAddr, StartupError> {
    let ip: IpAddr = host.parse().map_err(|_| StartupError::Bind {
        address: format!("{host}:{port}"),
    })?;
    Ok(SocketAddr::new(ip, port))
}

#[cfg(test)]
mod tests {
    use super::{
        Engine, Loaded, bind_listener, health_router, named_state, routes, serve_addr, serve_router,
    };
    use crate::generate::{Model, ModelError, TokenRow};
    use crate::load::StartupError;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn get_health_is_ok() {
        let response = health_router()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(bytes.as_ref(), br#"{"status":"ok"}"#);
    }

    #[tokio::test]
    async fn get_models_lists_the_loaded_name() {
        for name in ["tiny", "other name"] {
            let response = serve_router(name)
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let expected = format!(
                r#"{{"object":"list","data":[{{"id":"{name}","object":"model","owned_by":"local"}}]}}"#
            );
            assert_eq!(bytes.as_ref(), expected.as_bytes());
        }
    }

    #[tokio::test]
    async fn empty_messages_are_a_bad_request() {
        let response = serve_router("tiny")
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"tiny","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"messages must not be empty","type":"invalid_request_error","code":"bad_request"}}"#
        );
    }

    #[tokio::test]
    async fn an_unknown_role_is_a_bad_request() {
        for body in [
            r#"{"model":"tiny","messages":[{"role":"tool","content":"hi"}]}"#,
            r#"{"model":"tiny","messages":[{"role":"user","content":"hi"},{"role":"tool","content":"x"}]}"#,
        ] {
            let response = serve_router("tiny")
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{body}");
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                bytes.as_ref(),
                br#"{"error":{"message":"the message role is unknown","type":"invalid_request_error","code":"bad_request"}}"#,
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn a_non_string_content_is_a_bad_request() {
        for body in [
            r#"{"model":"tiny","messages":[{"role":"user","content":1}]}"#,
            r#"{"model":"tiny","messages":[{"role":"user","content":null}]}"#,
            r#"{"model":"tiny","messages":[{"role":"user","content":["hi"]}]}"#,
            r#"{"model":"tiny","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":{"text":"x"}}]}"#,
        ] {
            let response = serve_router("tiny")
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{body}");
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                bytes.as_ref(),
                br#"{"error":{"message":"message content must be a string","type":"invalid_request_error","code":"bad_request"}}"#,
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn a_different_model_name_is_not_found() {
        let response = serve_router("tiny")
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the model was not found","type":"invalid_request_error","code":"model_not_found"}}"#
        );
    }

    #[tokio::test]
    async fn a_known_role_is_not_rejected() {
        for (role, content) in [("system", "hi"), ("user", "hi"), ("assistant", "")] {
            let body = format!(
                r#"{{"model":"tiny","messages":[{{"role":"{role}","content":"{content}"}}]}}"#
            );
            let response = serve_router("tiny")
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 500, "{role}");
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                bytes.as_ref(),
                br#"{"error":{"message":"the completion is not available","type":"server_error","code":"internal"}}"#,
                "{role}"
            );
        }
    }

    #[tokio::test]
    async fn a_wrong_method_on_a_known_path_is_method_not_allowed() {
        for (method, uri) in [
            ("POST", "/health"),
            ("POST", "/v1/models"),
            ("GET", "/v1/chat/completions"),
        ] {
            let response = serve_router("tiny")
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 405, "{method} {uri}");
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                bytes.as_ref(),
                br#"{"error":{"message":"method not allowed","type":"invalid_request_error","code":"method_not_allowed"}}"#,
                "{uri}"
            );
        }
    }

    #[tokio::test]
    async fn an_unknown_path_is_not_method_not_allowed() {
        let response = serve_router("tiny")
            .oneshot(
                Request::builder()
                    .uri("/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_health_over_the_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let std_listener = bind_listener("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = std_listener.local_addr().unwrap().port();
        std_listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, health_router()).await;
        });
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).await.unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 "));
        assert!(text.contains("\r\n\r\n{\"status\":\"ok\"}"));
    }

    #[test]
    fn binding_a_taken_port_names_the_port() {
        let first = bind_listener("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = first.local_addr().unwrap().port();
        let error = bind_listener(([127, 0, 0, 1], port).into()).unwrap_err();
        assert_eq!(error, StartupError::PortInUse { port });
        assert_eq!(error.to_string(), format!("the port {port} is in use"));
    }

    #[test]
    fn a_host_that_is_not_an_address_is_not_in_use() {
        let error = serve_addr("not-a-host", 8080).unwrap_err();
        assert_eq!(
            error,
            StartupError::Bind {
                address: "not-a-host:8080".to_string(),
            }
        );
        assert_eq!(error.to_string(), "failed to bind not-a-host:8080");
    }

    #[derive(Clone)]
    struct Hold {
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
    }

    struct Script {
        rows: Vec<Vec<f32>>,
        cursor: usize,
        hold: Option<Hold>,
    }

    impl Model for Script {
        fn forward(&mut self, batch: &[TokenRow]) -> Result<Vec<Vec<f32>>, ModelError> {
            if let Some(hold) = &self.hold {
                hold.entered.store(true, Ordering::Relaxed);
                while !hold.release.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            let row = self.rows[self.cursor].clone();
            self.cursor += 1;
            Ok(batch.iter().map(|_| row.clone()).collect())
        }
    }

    fn hold_pair() -> Hold {
        Hold {
            entered: Arc::new(AtomicBool::new(false)),
            release: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn wait_until_entered(hold: &Hold) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !hold.entered.load(Ordering::Relaxed) {
            assert!(Instant::now() < deadline, "the model did not start");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn chat_vocab() -> Vec<String> {
        [
            "<|im_start|>",
            "user",
            "\n",
            "hi",
            "<|im_end|>",
            "assistant",
            "yo",
            "</s>",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    fn yo_rows() -> Vec<Vec<f32>> {
        let mut reply = vec![0.0; 8];
        reply[6] = 1.0;
        let mut eos = vec![0.0; 8];
        eos[7] = 1.0;
        vec![reply, eos]
    }

    fn scored(
        context: u32,
        queue: usize,
        key: Option<&str>,
        body_limit: usize,
        model: Script,
    ) -> axum::Router {
        let engine = Engine {
            loaded: Loaded {
                vocab: chat_vocab(),
                eos_id: 7,
                add_bos: false,
                bos_id: 0,
                working_context: context,
                template: None,
            },
            model: std::sync::Mutex::new(Box::new(model)),
        };
        routes(named_state(
            "tiny",
            Some(Arc::new(engine)),
            key.map(str::to_string),
            body_limit,
            queue,
        ))
    }

    fn chat_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn a_greedy_chat_returns_one_completion() {
        let app = scored(
            64,
            64,
            None,
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: None,
            },
        );
        let response = app
            .oneshot(chat_request(
                r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "tiny");
        assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(body["choices"][0]["message"]["content"], "yo");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 9);
        assert_eq!(body["usage"]["completion_tokens"], 1);
        assert_eq!(body["usage"]["total_tokens"], 10);
    }

    #[tokio::test]
    async fn a_stream_ends_with_done() {
        let app = scored(
            64,
            64,
            None,
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: None,
            },
        );
        let response = app
            .oneshot(chat_request(
                r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0,"stream":true}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("yo"));
        assert!(text.contains("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn a_prompt_past_the_context_is_400() {
        let app = scored(
            1,
            64,
            None,
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: None,
            },
        );
        let response = app
            .oneshot(chat_request(
                r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the prompt is longer than the working context","type":"invalid_request_error","code":"context_length_exceeded"}}"#
        );
    }

    #[tokio::test]
    async fn a_missing_api_key_is_rejected_and_health_stays_open() {
        let app = scored(
            64,
            64,
            Some("secret"),
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: None,
            },
        );
        let missing = app
            .clone()
            .oneshot(chat_request(
                r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(missing.status(), 401);
        let bytes = missing.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the api key is invalid","type":"authentication_error","code":"invalid_api_key"}}"#
        );
        let mut authed = chat_request(
            r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0}"#,
        );
        authed
            .headers_mut()
            .insert("authorization", "Bearer secret".parse().unwrap());
        let ok = app.clone().oneshot(authed).await.unwrap();
        assert_eq!(ok.status(), 200);
        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), 200);
    }

    #[tokio::test]
    async fn a_body_over_the_limit_is_413() {
        let app = scored(
            64,
            64,
            None,
            8,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: None,
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("content-length", "64")
            .body(Body::from(r#"{"model":"tiny","messages":[]}"#))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), 413);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the request body is too large","type":"invalid_request_error","code":"body_too_large"}}"#
        );
    }

    #[tokio::test]
    async fn a_full_queue_is_503_while_health_stays_up() {
        let hold = hold_pair();
        let app = scored(
            64,
            0,
            None,
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: Some(hold.clone()),
            },
        );
        let first = app.clone();
        let running = tokio::spawn(async move {
            first
                .oneshot(chat_request(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0}"#,
                ))
                .await
                .unwrap()
        });
        wait_until_entered(&hold).await;
        let full = app
            .clone()
            .oneshot(chat_request(
                r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(full.status(), 503);
        let bytes = full.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the queue is full","type":"server_error","code":"queue_full"}}"#
        );
        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), 200);
        hold.release.store(true, Ordering::Relaxed);
        assert_eq!(running.await.unwrap().status(), 200);
    }

    #[tokio::test]
    async fn a_dropped_chat_leaves_health_ok() {
        let hold = hold_pair();
        let app = scored(
            64,
            64,
            None,
            1024 * 1024,
            Script {
                rows: yo_rows(),
                cursor: 0,
                hold: Some(hold.clone()),
            },
        );
        let first = app.clone();
        let running = tokio::spawn(async move {
            first
                .oneshot(chat_request(
                    r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0}"#,
                ))
                .await
                .unwrap()
        });
        wait_until_entered(&hold).await;
        running.abort();
        let health = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), 200);
        hold.release.store(true, Ordering::Relaxed);
        let _ = running.await;
    }
}
