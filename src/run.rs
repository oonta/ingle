use std::ffi::OsString;

use crate::cli::{parse_args, read_prompt, Command, GenerateCommand, UsageError};
use crate::encode::{decode_tokens, encode_prompt, DecodeError, EncodeError};
use crate::generate::{greedy_continue, GenerateError, Model};
use crate::llama::load_llama;
use crate::load::{ModelCard, StartupError};
use crate::platform::{detect_ram_mb, require_host_simd, resolve_kv_pool, startup_line, target_triple, PoolRequest};

/// Working context when `generate` is not given `--max-context`.
const DEFAULT_MAX_CONTEXT: u32 = 4096;

/// `ingle generate` could not print a continuation.
#[derive(Debug)]
pub enum RunError {
    Usage(UsageError),
    EmptyPrompt,
    Startup(StartupError),
    Continuation(ContinuationError),
}

/// Parse `args`, load the GGUF file, and return the greedy continuation.
/// `stdin` is used only when the positional prompt is omitted.
pub fn generate_text<I, S>(args: I, infer_model: Option<&str>, stdin: &str) -> Result<String, RunError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString> + Clone,
{
    let command = parse_args(args, infer_model).map_err(RunError::Usage)?;
    let Command::Generate(command) = command;
    generate_from_command(command, stdin)
}

/// Score an already parsed `generate` command. The caller reads stdin only for [`crate::cli::PromptInput::Stdin`].
pub fn generate_from_command(command: GenerateCommand, stdin: &str) -> Result<String, RunError> {
    let prompt = read_prompt(command.prompt, stdin).map_err(|_| RunError::EmptyPrompt)?;
    require_host_simd().map_err(RunError::Startup)?;
    let pool = resolve_kv_pool(PoolRequest::Auto, detect_ram_mb()).map_err(RunError::Startup)?;
    tracing::info!("{}", startup_line(target_triple(), &pool));
    let (card, mut model) =
        load_llama(&command.model, DEFAULT_MAX_CONTEXT).map_err(RunError::Startup)?;
    continuation_text(&mut model, &card, &prompt, command.max_tokens)
        .map_err(RunError::Continuation)
}

/// The prompt did not become printed text.
#[derive(Debug, PartialEq, Eq)]
pub enum ContinuationError {
    Encode(EncodeError),
    Generate(GenerateError),
    Decode(DecodeError),
}

/// Encode `prompt`, score it, and decode the new tokens. EOS is already omitted.
pub fn continuation_text<M: Model>(
    model: &mut M,
    card: &ModelCard,
    prompt: &str,
    max_new_tokens: u32,
) -> Result<String, ContinuationError> {
    let prompt_ids = encode_prompt(prompt, &card.tokens, card.add_bos, card.bos_id)
        .map_err(ContinuationError::Encode)?;
    let generated = greedy_continue(model, &prompt_ids, max_new_tokens, card.eos_id)
        .map_err(ContinuationError::Generate)?;
    decode_tokens(&generated.tokens, &card.tokens).map_err(ContinuationError::Decode)
}

#[cfg(test)]
mod tests {
    use super::{continuation_text, generate_text, ContinuationError, RunError};
    use crate::generate::{Model, ModelError, TokenRow};
    use crate::load::{ModelCard, StartupError};

    struct Script {
        rows: Vec<Vec<f32>>,
        cursor: usize,
    }

    impl Model for Script {
        fn forward(&mut self, batch: &[TokenRow]) -> Result<Vec<Vec<f32>>, ModelError> {
            let mut out = Vec::with_capacity(batch.len());
            for _ in batch {
                let row = self.rows.get(self.cursor).cloned().ok_or(ModelError)?;
                self.cursor += 1;
                out.push(row);
            }
            Ok(out)
        }
    }

    fn card() -> ModelCard {
        ModelCard {
            name: "toy".to_string(),
            context_length: 128,
            working_context: 128,
            block_count: 1,
            head_count: 1,
            head_count_kv: 1,
            embedding_length: 8,
            head_dim: 8,
            rope_freq_base: 10_000.0,
            eos_id: 2,
            add_bos: true,
            bos_id: 0,
            vocab_size: 3,
            tokens: ["<s>", "yo", "</s>"].map(str::to_string).to_vec(),
        }
    }

    #[test]
    fn a_scripted_score_prints_the_decoded_piece() {
        let mut model = Script {
            rows: vec![
                vec![0.0, 0.0, 1.0],
                vec![0.0, 4.0, 0.0],
                vec![0.0, 0.0, 9.0],
            ],
            cursor: 0,
        };
        let text = continuation_text(&mut model, &card(), "yo", 8).unwrap();
        assert_eq!(text, "yo");
        assert_eq!(model.cursor, 3);
    }

    #[test]
    fn an_unencodable_prompt_is_not_scored() {
        let mut model = Script {
            rows: Vec::new(),
            cursor: 0,
        };
        let error = continuation_text(&mut model, &card(), "Z", 8).unwrap_err();
        assert!(matches!(error, ContinuationError::Encode(_)));
        assert_eq!(model.cursor, 0);
    }

    #[test]
    fn an_empty_prompt_does_not_open_a_model_file() {
        let error = generate_text(
            ["ingle", "generate", "--model", "missing.gguf", ""],
            None,
            "from stdin",
        )
        .unwrap_err();
        assert!(matches!(error, RunError::EmptyPrompt));
    }

    #[test]
    fn a_missing_model_file_is_a_startup_error() {
        let path = std::env::temp_dir().join("ingle-no-such-model.gguf");
        let _ = std::fs::remove_file(&path);
        let error = generate_text(
            [
                "ingle",
                "generate",
                "--model",
                path.to_str().expect("utf-8 temp path"),
                "hi",
            ],
            None,
            "",
        )
        .unwrap_err();
        match error {
            RunError::Startup(StartupError::FileNotFound { path: got }) => assert_eq!(got, path),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn an_unknown_flag_is_usage() {
        let error = generate_text(["ingle", "generate", "--nope"], None, "").unwrap_err();
        match error {
            RunError::Usage(error) => assert!(error.message.contains("--nope")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
