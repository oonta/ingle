use std::ffi::OsString;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};

/// A parsed invocation of the `ingle` binary.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Generate(GenerateCommand),
    Serve(ServeCommand),
}

/// Arguments of `ingle generate` after defaults and `INFER_MODEL` are applied.
#[derive(Debug, PartialEq, Eq)]
pub struct GenerateCommand {
    pub model: PathBuf,
    pub max_tokens: u32,
    pub prompt: PromptInput,
}

/// Arguments of `ingle serve` after defaults and `INFER_MODEL` are applied.
#[derive(Debug, PartialEq, Eq)]
pub struct ServeCommand {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub max_body_bytes: usize,
    pub queue_capacity: usize,
    pub shutdown_timeout_ms: u64,
}

/// Environment values that fill serve flags the command line omitted.
#[derive(Debug, Default, Clone, Copy)]
pub struct ServeEnv<'a> {
    pub api_key: Option<&'a str>,
    pub max_body_bytes: Option<&'a str>,
    pub queue_capacity: Option<&'a str>,
    pub shutdown_timeout_ms: Option<&'a str>,
}

/// Where the prompt text comes from.
#[derive(Debug, PartialEq, Eq)]
pub enum PromptInput {
    /// The positional argument, even when it is empty.
    Text(String),
    /// The positional argument was omitted.
    Stdin,
}

/// The process arguments are not a valid command.
#[derive(Debug, PartialEq, Eq)]
pub struct UsageError {
    pub message: String,
}

/// The prompt that would be tokenized is empty.
#[derive(Debug, PartialEq, Eq)]
pub struct EmptyPrompt;

#[derive(Parser)]
#[command(name = "ingle")]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Print a greedy continuation of the prompt.
    Generate {
        /// Path to the GGUF file. Overrides `INFER_MODEL`.
        #[arg(long, value_name = "PATH")]
        model: Option<PathBuf>,
        /// Maximum number of new tokens, from 1 through 4096.
        #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..=4096))]
        max_tokens: u32,
        /// Prompt text. When omitted, the prompt is read from stdin.
        prompt: Option<String>,
    },
    /// Serve the loaded model over HTTP.
    Serve {
        /// Path to the GGUF file. Overrides `INFER_MODEL`.
        #[arg(long, value_name = "PATH")]
        model: Option<PathBuf>,
        /// Address to bind. Defaults to the loopback interface.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Bearer token. An empty value disables the check.
        #[arg(long, value_name = "KEY")]
        api_key: Option<String>,
        /// Maximum request body in bytes.
        #[arg(long, default_value_t = 1_048_576)]
        max_body_bytes: usize,
        /// Requests that may wait behind the one that is running.
        #[arg(long, default_value_t = 64)]
        queue_capacity: usize,
        /// Milliseconds to drain the current request after Ctrl+C.
        #[arg(long, default_value_t = 10_000)]
        shutdown_timeout_ms: u64,
    },
}

/// Parse `args` as `ingle` saw them, including the program name.
/// `infer_model` is the value of `INFER_MODEL`. A `--model` flag wins.
pub fn parse_args<I, S>(args: I, infer_model: Option<&str>) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            return Err(UsageError {
                message: error.to_string(),
            });
        }
    };
    match cli.command {
        CliCommand::Generate {
            model,
            max_tokens,
            prompt,
        } => {
            let model = resolve_model(model, infer_model)?;
            let prompt = match prompt {
                Some(text) => PromptInput::Text(text),
                None => PromptInput::Stdin,
            };
            Ok(Command::Generate(GenerateCommand {
                model,
                max_tokens,
                prompt,
            }))
        }
        CliCommand::Serve {
            model,
            host,
            port,
            api_key,
            max_body_bytes,
            queue_capacity,
            shutdown_timeout_ms,
        } => {
            let model = resolve_model(model, infer_model)?;
            Ok(Command::Serve(ServeCommand {
                model,
                host,
                port,
                api_key,
                max_body_bytes,
                queue_capacity,
                shutdown_timeout_ms,
            }))
        }
    }
}

fn resolve_model(model: Option<PathBuf>, infer_model: Option<&str>) -> Result<PathBuf, UsageError> {
    match model {
        Some(path) => Ok(path),
        None => match infer_model {
            Some(path) if !path.is_empty() => Ok(PathBuf::from(path)),
            _ => Err(UsageError {
                message: format!(
                    "missing --model (or INFER_MODEL)\n\n{}",
                    Cli::command().render_help()
                ),
            }),
        },
    }
}

/// Fill serve limits the command line left at their defaults.
/// A present flag wins. An empty `INFER_API_KEY` or an empty `--api-key` disables the check.
pub fn apply_serve_env(
    command: &mut ServeCommand,
    args: &[impl AsRef<str>],
    env: ServeEnv<'_>,
) -> Result<(), UsageError> {
    if !flag_set(args, "--api-key") {
        if let Some(value) = nonempty(env.api_key) {
            command.api_key = Some(value.to_string());
        }
    } else if command.api_key.as_deref().is_some_and(str::is_empty) {
        command.api_key = None;
    }
    if !flag_set(args, "--max-body-bytes") {
        if let Some(value) = nonempty(env.max_body_bytes) {
            command.max_body_bytes = parse_env(value, "INFER_MAX_BODY_BYTES")?;
        }
    }
    if !flag_set(args, "--queue-capacity") {
        if let Some(value) = nonempty(env.queue_capacity) {
            command.queue_capacity = parse_env(value, "INFER_QUEUE_CAPACITY")?;
        }
    }
    if !flag_set(args, "--shutdown-timeout-ms") {
        if let Some(value) = nonempty(env.shutdown_timeout_ms) {
            command.shutdown_timeout_ms = parse_env(value, "INFER_SHUTDOWN_TIMEOUT_MS")?;
        }
    }
    Ok(())
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|text| !text.is_empty())
}

fn flag_set(args: &[impl AsRef<str>], name: &str) -> bool {
    let with_equals = format!("{name}=");
    args.iter().any(|arg| {
        let arg = arg.as_ref();
        arg == name || arg.starts_with(&with_equals)
    })
}

fn parse_env<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, UsageError> {
    value.parse().map_err(|_| UsageError {
        message: format!(
            "{name} is not an integer\n\n{}",
            Cli::command().render_help()
        ),
    })
}

/// Resolve the prompt text. A positional value is used as-is and `stdin` is ignored.
pub fn read_prompt(input: PromptInput, stdin: &str) -> Result<String, EmptyPrompt> {
    let text = match input {
        PromptInput::Text(text) => text,
        PromptInput::Stdin => stdin.to_string(),
    };
    if text.is_empty() {
        Err(EmptyPrompt)
    } else {
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Command, EmptyPrompt, GenerateCommand, PromptInput, ServeCommand, ServeEnv,
        apply_serve_env, parse_args, read_prompt,
    };
    use std::path::PathBuf;

    fn generate<'a>(
        args: impl IntoIterator<Item = &'a str>,
        infer_model: Option<&str>,
    ) -> GenerateCommand {
        match parse_args(args, infer_model).unwrap() {
            Command::Generate(command) => command,
            Command::Serve(_) => panic!("expected generate"),
        }
    }

    fn serve<'a>(
        args: impl IntoIterator<Item = &'a str>,
        infer_model: Option<&str>,
    ) -> ServeCommand {
        match parse_args(args, infer_model).unwrap() {
            Command::Serve(command) => command,
            Command::Generate(_) => panic!("expected serve"),
        }
    }

    #[test]
    fn model_flag_overrides_the_environment() {
        let command = generate(
            ["ingle", "generate", "--model", "from-flag.gguf", "hello"],
            Some("from-env.gguf"),
        );
        assert_eq!(
            command,
            GenerateCommand {
                model: PathBuf::from("from-flag.gguf"),
                max_tokens: 256,
                prompt: PromptInput::Text("hello".to_string()),
            }
        );
    }

    #[test]
    fn environment_supplies_the_model_when_the_flag_is_omitted() {
        let command = generate(["ingle", "generate"], Some("from-env.gguf"));
        assert_eq!(command.model, PathBuf::from("from-env.gguf"));
        assert_eq!(command.prompt, PromptInput::Stdin);
        assert_eq!(command.max_tokens, 256);
    }

    #[test]
    fn missing_model_is_a_usage_error() {
        let error = parse_args(["ingle", "generate", "hello"], None).unwrap_err();
        assert!(error.message.contains("--model"));
    }

    #[test]
    fn max_tokens_accepts_the_closed_range() {
        let low = generate(
            [
                "ingle",
                "generate",
                "--model",
                "m.gguf",
                "--max-tokens",
                "1",
            ],
            None,
        );
        let high = generate(
            [
                "ingle",
                "generate",
                "--model",
                "m.gguf",
                "--max-tokens",
                "4096",
            ],
            None,
        );
        assert_eq!(low.max_tokens, 1);
        assert_eq!(high.max_tokens, 4096);
    }

    #[test]
    fn max_tokens_outside_the_range_is_a_usage_error() {
        for raw in ["0", "4097"] {
            let error = parse_args(
                [
                    "ingle",
                    "generate",
                    "--model",
                    "m.gguf",
                    "--max-tokens",
                    raw,
                ],
                None,
            )
            .unwrap_err();
            assert!(error.message.contains("max-tokens"), "{raw}");
        }
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error() {
        let error =
            parse_args(["ingle", "generate", "--model", "m.gguf", "--nope"], None).unwrap_err();
        assert!(error.message.contains("--nope"));
    }

    #[test]
    fn a_positional_prompt_is_not_replaced_by_stdin() {
        let command = generate(["ingle", "generate", "--model", "m.gguf", "hi"], None);
        assert_eq!(read_prompt(command.prompt, "from stdin").unwrap(), "hi");
    }

    #[test]
    fn an_omitted_positional_reads_stdin() {
        let command = generate(["ingle", "generate", "--model", "m.gguf"], None);
        assert_eq!(
            read_prompt(command.prompt, "from stdin").unwrap(),
            "from stdin"
        );
    }

    #[test]
    fn an_empty_positional_prompt_is_rejected() {
        let command = generate(["ingle", "generate", "--model", "m.gguf", ""], None);
        assert_eq!(read_prompt(command.prompt, "from stdin"), Err(EmptyPrompt));
    }

    #[test]
    fn serve_defaults_to_loopback_8080() {
        let command = serve(["ingle", "serve", "--model", "m.gguf"], None);
        assert_eq!(
            command,
            ServeCommand {
                model: PathBuf::from("m.gguf"),
                host: "127.0.0.1".to_string(),
                port: 8080,
                api_key: None,
                max_body_bytes: 1_048_576,
                queue_capacity: 64,
                shutdown_timeout_ms: 10_000,
            }
        );
    }

    #[test]
    fn serve_host_and_port_replace_the_defaults() {
        let command = serve(
            [
                "ingle", "serve", "--model", "m.gguf", "--host", "0.0.0.0", "--port", "9",
            ],
            None,
        );
        assert_eq!(command.host, "0.0.0.0");
        assert_eq!(command.port, 9);
    }

    #[test]
    fn serve_uses_the_environment_when_the_flag_is_omitted() {
        let command = serve(["ingle", "serve"], Some("from-env.gguf"));
        assert_eq!(command.model, PathBuf::from("from-env.gguf"));
        assert_eq!(command.port, 8080);
    }

    #[test]
    fn an_api_key_from_the_environment_fills_an_omitted_flag() {
        let mut command = serve(["ingle", "serve", "--model", "m.gguf"], None);
        apply_serve_env(
            &mut command,
            &["ingle", "serve", "--model", "m.gguf"],
            ServeEnv {
                api_key: Some("secret"),
                max_body_bytes: Some("32"),
                queue_capacity: Some("2"),
                shutdown_timeout_ms: Some("50"),
            },
        )
        .unwrap();
        assert_eq!(command.api_key.as_deref(), Some("secret"));
        assert_eq!(command.max_body_bytes, 32);
        assert_eq!(command.queue_capacity, 2);
        assert_eq!(command.shutdown_timeout_ms, 50);
    }

    #[test]
    fn a_present_flag_wins_over_the_environment() {
        let mut command = serve(
            [
                "ingle",
                "serve",
                "--model",
                "m.gguf",
                "--api-key",
                "flag",
                "--queue-capacity",
                "3",
            ],
            None,
        );
        apply_serve_env(
            &mut command,
            &[
                "ingle",
                "serve",
                "--model",
                "m.gguf",
                "--api-key",
                "flag",
                "--queue-capacity",
                "3",
            ],
            ServeEnv {
                api_key: Some("env"),
                queue_capacity: Some("9"),
                ..ServeEnv::default()
            },
        )
        .unwrap();
        assert_eq!(command.api_key.as_deref(), Some("flag"));
        assert_eq!(command.queue_capacity, 3);
        assert_eq!(command.max_body_bytes, 1_048_576);
    }

    #[test]
    fn an_empty_api_key_disables_the_check() {
        let mut command = serve(
            ["ingle", "serve", "--model", "m.gguf", "--api-key", ""],
            None,
        );
        apply_serve_env(
            &mut command,
            &["ingle", "serve", "--model", "m.gguf", "--api-key", ""],
            ServeEnv {
                api_key: Some("env"),
                ..ServeEnv::default()
            },
        )
        .unwrap();
        assert_eq!(command.api_key, None);
    }

    #[test]
    fn a_bad_limit_in_the_environment_is_a_usage_error() {
        let mut command = serve(["ingle", "serve", "--model", "m.gguf"], None);
        let error = apply_serve_env(
            &mut command,
            &["ingle", "serve", "--model", "m.gguf"],
            ServeEnv {
                queue_capacity: Some("nope"),
                ..ServeEnv::default()
            },
        )
        .unwrap_err();
        assert!(error.message.contains("INFER_QUEUE_CAPACITY"));
    }

    #[test]
    fn serve_without_a_model_is_a_usage_error() {
        let error = parse_args(["ingle", "serve"], None).unwrap_err();
        assert!(error.message.contains("--model"));
    }

    #[test]
    fn empty_stdin_is_rejected() {
        let command = generate(["ingle", "generate", "--model", "m.gguf"], None);
        assert_eq!(read_prompt(command.prompt, ""), Err(EmptyPrompt));
    }
}
