use std::ffi::OsString;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};

/// A parsed invocation of the `ingle` binary.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Generate(GenerateCommand),
}

/// Arguments of `ingle generate` after defaults and `INFER_MODEL` are applied.
#[derive(Debug, PartialEq, Eq)]
pub struct GenerateCommand {
    pub model: PathBuf,
    pub max_tokens: u32,
    pub prompt: PromptInput,
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
    let CliCommand::Generate {
        model,
        max_tokens,
        prompt,
    } = cli.command;
    let model = match model {
        Some(path) => path,
        None => match infer_model {
            Some(path) if !path.is_empty() => PathBuf::from(path),
            _ => {
                return Err(UsageError {
                    message: format!("missing --model (or INFER_MODEL)\n\n{}", Cli::command().render_help()),
                });
            }
        },
    };
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
    use super::{Command, EmptyPrompt, GenerateCommand, PromptInput, parse_args, read_prompt};
    use std::path::PathBuf;

    fn generate<'a>(args: impl IntoIterator<Item = &'a str>, infer_model: Option<&str>) -> GenerateCommand {
        match parse_args(args, infer_model).unwrap() {
            Command::Generate(command) => command,
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
            ["ingle", "generate", "--model", "m.gguf", "--max-tokens", "1"],
            None,
        );
        let high = generate(
            ["ingle", "generate", "--model", "m.gguf", "--max-tokens", "4096"],
            None,
        );
        assert_eq!(low.max_tokens, 1);
        assert_eq!(high.max_tokens, 4096);
    }

    #[test]
    fn max_tokens_outside_the_range_is_a_usage_error() {
        for raw in ["0", "4097"] {
            let error = parse_args(
                ["ingle", "generate", "--model", "m.gguf", "--max-tokens", raw],
                None,
            )
            .unwrap_err();
            assert!(error.message.contains("max-tokens"), "{raw}");
        }
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error() {
        let error = parse_args(
            ["ingle", "generate", "--model", "m.gguf", "--nope"],
            None,
        )
        .unwrap_err();
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
    fn empty_stdin_is_rejected() {
        let command = generate(["ingle", "generate", "--model", "m.gguf"], None);
        assert_eq!(read_prompt(command.prompt, ""), Err(EmptyPrompt));
    }
}
