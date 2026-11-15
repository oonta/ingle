use minijinja::{Environment, context};
use serde::Serialize;
use serde_json::{Map, Value};

/// One accepted chat after the body has passed the stage 1 checks.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedChat {
    pub messages: Vec<(String, String)>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub max_tokens: u32,
    pub stream: bool,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

/// The body cannot become a completion.
#[derive(Debug, PartialEq, Eq)]
pub struct RejectedChat {
    pub status: u16,
    pub message: &'static str,
    pub code: &'static str,
}

/// ChatML prompt used when the GGUF file has no chat template.
pub fn chatml_prompt(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// The Jinja template did not render.
#[derive(Debug)]
pub struct TemplateError;

/// Render a Jinja chat template. `add_generation_prompt` is true.
pub fn render_template(
    template: &str,
    messages: &[(String, String)],
) -> Result<String, TemplateError> {
    let mut environment = Environment::new();
    environment
        .add_template("chat", template)
        .map_err(|error| {
            tracing::error!("the chat template failed: {error}");
            TemplateError
        })?;
    let messages: Vec<TemplateMessage> = messages
        .iter()
        .map(|(role, content)| TemplateMessage { role, content })
        .collect();
    environment
        .get_template("chat")
        .and_then(|template| {
            template.render(context! {
                messages => messages,
                add_generation_prompt => true,
            })
        })
        .map_err(|error| {
            tracing::error!("the chat template failed: {error}");
            TemplateError
        })
}

#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Check `body` against the loaded model name.
pub fn accept_chat(body: &Value, model_name: &str) -> Result<AcceptedChat, RejectedChat> {
    let object = body
        .as_object()
        .ok_or(invalid("the request body is invalid"))?;
    reject_later_version(object)?;
    let messages = messages(object)?;
    let temperature = number(
        object,
        "temperature",
        1.0,
        0.0,
        2.0,
        "temperature must be from 0 through 2",
    )?;
    let greedy = temperature == 0.0;
    let top_p = if greedy {
        1.0
    } else {
        number(
            object,
            "top_p",
            1.0,
            f64::MIN_POSITIVE,
            1.0,
            "top_p must be greater than 0 through 1",
        )?
    };
    if !greedy && top_p <= 0.0 {
        return Err(invalid("top_p must be greater than 0 through 1"));
    }
    let top_k = optional_u32(object, "top_k", 1, 200, "top_k must be from 1 through 200")?;
    let max_tokens = max_tokens(object)?;
    let stream = optional_bool(object, "stream")?.unwrap_or(false);
    let stop = stop_strings(object)?;
    let seed = seed(object)?;
    let frequency_penalty = number(
        object,
        "frequency_penalty",
        0.0,
        -2.0,
        2.0,
        "frequency_penalty must be from -2 through 2",
    )?;
    let presence_penalty = number(
        object,
        "presence_penalty",
        0.0,
        -2.0,
        2.0,
        "presence_penalty must be from -2 through 2",
    )?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or(invalid("the request body is invalid"))?;
    if model != model_name {
        return Err(RejectedChat {
            status: 404,
            message: "the model was not found",
            code: "model_not_found",
        });
    }
    Ok(AcceptedChat {
        messages,
        temperature: temperature as f32,
        top_p: top_p as f32,
        top_k,
        max_tokens,
        stream,
        stop,
        seed,
        frequency_penalty: frequency_penalty as f32,
        presence_penalty: presence_penalty as f32,
    })
}

fn invalid(message: &'static str) -> RejectedChat {
    RejectedChat {
        status: 400,
        message,
        code: "bad_request",
    }
}

fn reject_later_version(object: &Map<String, Value>) -> Result<(), RejectedChat> {
    for key in ["tools", "tool_choice", "response_format"] {
        if object.contains_key(key) {
            return Err(later(key));
        }
    }
    if object.get("logprobs").and_then(Value::as_bool) == Some(true) {
        return Err(later("logprobs"));
    }
    if object
        .get("n")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return Err(later("n"));
    }
    if let Some(value) = object.get("n") {
        if value.as_u64() != Some(1) {
            return Err(invalid("n must be 1"));
        }
    }
    Ok(())
}

fn later(feature: &'static str) -> RejectedChat {
    match feature {
        "tools" => invalid("tools belongs to a later version"),
        "tool_choice" => invalid("tool_choice belongs to a later version"),
        "response_format" => invalid("response_format belongs to a later version"),
        "logprobs" => invalid("logprobs belongs to a later version"),
        "n" => invalid("n belongs to a later version"),
        _ => invalid("the feature belongs to a later version"),
    }
}

fn messages(object: &Map<String, Value>) -> Result<Vec<(String, String)>, RejectedChat> {
    let values = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(invalid("the request body is invalid"))?;
    if values.is_empty() {
        return Err(invalid("messages must not be empty"));
    }
    if values.len() > 256 {
        return Err(invalid("messages must contain at most 256 items"));
    }
    let mut messages = Vec::with_capacity(values.len());
    for value in values {
        let item = value
            .as_object()
            .ok_or(invalid("the request body is invalid"))?;
        let role = item
            .get("role")
            .and_then(Value::as_str)
            .ok_or(invalid("the request body is invalid"))?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(invalid("the message role is unknown"));
        }
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .ok_or(invalid("message content must be a string"))?;
        messages.push((role.to_string(), content.to_string()));
    }
    Ok(messages)
}

fn max_tokens(object: &Map<String, Value>) -> Result<u32, RejectedChat> {
    let max_tokens = object.get("max_tokens");
    let completion = object.get("max_completion_tokens");
    if max_tokens.is_some() && completion.is_some() {
        return Err(invalid(
            "max_tokens and max_completion_tokens cannot both be set",
        ));
    }
    let value = completion.or(max_tokens);
    match value {
        None => Ok(256),
        Some(value) => {
            let number = value
                .as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or(invalid("max_tokens must be from 1 through 4096"))?;
            if (1..=4096).contains(&number) {
                Ok(number)
            } else {
                Err(invalid("max_tokens must be from 1 through 4096"))
            }
        }
    }
}

fn number(
    object: &Map<String, Value>,
    key: &str,
    default: f64,
    low: f64,
    high: f64,
    message: &'static str,
) -> Result<f64, RejectedChat> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => {
            let number = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or(invalid(message))?;
            if number < low || number > high {
                Err(invalid(message))
            } else {
                Ok(number)
            }
        }
    }
}

fn optional_u32(
    object: &Map<String, Value>,
    key: &str,
    low: u32,
    high: u32,
    message: &'static str,
) -> Result<Option<u32>, RejectedChat> {
    match object.get(key) {
        None => Ok(None),
        Some(value) => {
            let number = value
                .as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .filter(|number| (low..=high).contains(number))
                .ok_or(invalid(message))?;
            Ok(Some(number))
        }
    }
}

fn optional_bool(object: &Map<String, Value>, key: &str) -> Result<Option<bool>, RejectedChat> {
    match object.get(key) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(invalid("the request body is invalid")),
    }
}

fn stop_strings(object: &Map<String, Value>) -> Result<Vec<String>, RejectedChat> {
    let Some(value) = object.get("stop") else {
        return Ok(Vec::new());
    };
    let stops = match value {
        Value::String(text) => vec![text.clone()],
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or(invalid("a stop string must be text"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(invalid("a stop string must be text")),
    };
    if stops.len() > 4 {
        return Err(invalid("at most 4 stop strings are allowed"));
    }
    if stops.iter().any(|stop| stop.len() > 64) {
        return Err(invalid("a stop string must be at most 64 bytes"));
    }
    Ok(stops)
}

fn seed(object: &Map<String, Value>) -> Result<Option<u64>, RejectedChat> {
    match object.get("seed") {
        None => Ok(None),
        Some(Value::Number(number)) => {
            if number.as_i64().is_some_and(|value| value < 0) || number.as_u64().is_none() {
                return Err(invalid("seed must be an unsigned integer"));
            }
            Ok(number.as_u64())
        }
        Some(_) => Err(invalid("seed must be an unsigned integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::{accept_chat, chatml_prompt, render_template};
    use serde_json::json;

    #[test]
    fn chatml_wraps_each_message_and_opens_the_assistant_turn() {
        let prompt = chatml_prompt(&[
            ("system".to_string(), "be brief".to_string()),
            ("user".to_string(), "hi".to_string()),
        ]);
        assert_eq!(
            prompt,
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_template_renders_messages_and_a_broken_template_fails() {
        let messages = vec![("user".to_string(), "hi".to_string())];
        let prompt = render_template("{{ messages[0].content }}", &messages).unwrap();
        assert_eq!(prompt, "hi");
        assert!(render_template("{{ missing | no_such_filter }}", &messages).is_err());
    }

    #[test]
    fn both_token_limits_are_rejected() {
        let error = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"max_completion_tokens":4}),
            "tiny",
        )
        .unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "bad_request");
        assert_eq!(
            error.message,
            "max_tokens and max_completion_tokens cannot both be set"
        );
    }

    #[test]
    fn tools_belong_to_a_later_version() {
        let error = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"tools":[]}),
            "tiny",
        )
        .unwrap_err();
        assert_eq!(error.message, "tools belongs to a later version");
    }

    #[test]
    fn max_completion_tokens_replaces_max_tokens() {
        let chat = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":4}),
            "tiny",
        )
        .unwrap();
        assert_eq!(chat.max_tokens, 4);
        assert!(!chat.stream);
        assert_eq!(chat.temperature, 1.0);
    }
}
