use std::io;
use std::net::{IpAddr, SocketAddr, TcpListener};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

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

#[derive(Deserialize)]
struct ChatMessage {}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
}

async fn chat(Json(request): Json<ChatRequest>) -> (StatusCode, Json<ErrorResponse>) {
    if request.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            "bad_request",
        );
    }
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "the completion is not available",
        "server_error",
        "internal",
    )
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
pub fn serve_router(model_name: &str) -> Router {
    Router::new()
        .route("/health", get(health).fallback(method_not_allowed))
        .route("/v1/models", get(list_models).fallback(method_not_allowed))
        .route(
            "/v1/chat/completions",
            post(chat).fallback(method_not_allowed),
        )
        .with_state(AppState {
            model_name: model_name.to_string(),
        })
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
    use super::{bind_listener, health_router, serve_addr, serve_router};
    use crate::load::StartupError;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
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
    async fn a_user_message_is_not_rejected_as_empty() {
        let response = serve_router("tiny")
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"tiny","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 500);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.as_ref(),
            br#"{"error":{"message":"the completion is not available","type":"server_error","code":"internal"}}"#
        );
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
}
