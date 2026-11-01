use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file::{Content, Value};

/// Metadata a later forward pass needs, taken from one GGUF file.
#[derive(Debug, PartialEq)]
pub struct ModelCard {
    pub name: String,
    pub context_length: u32,
    pub working_context: u32,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub embedding_length: u32,
    pub head_dim: u32,
    pub rope_freq_base: f32,
    pub eos_id: u32,
    pub add_bos: bool,
    pub bos_id: u32,
    pub vocab_size: u32,
}

/// A model file cannot be used. The text is the process's startup line.
#[derive(Debug, PartialEq, Eq)]
pub enum StartupError {
    FileNotFound { path: PathBuf },
    NotGguf { path: PathBuf },
    Architecture { path: PathBuf, found: String },
    UnsupportedQuantization { path: PathBuf },
    NoVocabulary { path: PathBuf },
    NoEos { path: PathBuf },
    MissingKey { path: PathBuf, key: &'static str },
    Read { path: PathBuf },
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound { path } => write!(f, "file not found: {}", path.display()),
            Self::NotGguf { path } => write!(f, "not a GGUF: {}", path.display()),
            Self::Architecture { path, found } => {
                write!(
                    f,
                    "the model architecture is not llama: found {found} in {}",
                    path.display()
                )
            }
            Self::UnsupportedQuantization { path } => {
                write!(f, "the quantization type is not supported: {}", path.display())
            }
            Self::NoVocabulary { path } => write!(f, "the file has no vocabulary: {}", path.display()),
            Self::NoEos { path } => {
                write!(f, "the file has no tokenizer.ggml.eos_token_id: {}", path.display())
            }
            Self::MissingKey { path, key } => write!(f, "missing {key} in {}", path.display()),
            Self::Read { path } => write!(f, "failed to read {}", path.display()),
        }
    }
}

/// Read the card from `path`. `max_context` caps `working_context`.
pub fn inspect_gguf(path: &Path, max_context: u32) -> Result<ModelCard, StartupError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(StartupError::FileNotFound {
                path: path.to_path_buf(),
            });
        }
        Err(_) => {
            return Err(StartupError::Read {
                path: path.to_path_buf(),
            });
        }
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() || (magic != *b"GGUF" && magic != *b"FUGG") {
        return Err(StartupError::NotGguf {
            path: path.to_path_buf(),
        });
    }
    file.seek(SeekFrom::Start(0)).map_err(|_| StartupError::Read {
        path: path.to_path_buf(),
    })?;
    let content = Content::read(&mut file).map_err(|error| classify_gguf_error(path, &error))?;

    let architecture = match content.metadata.get("general.architecture") {
        Some(Value::String(value)) => value.clone(),
        _ => {
            return Err(StartupError::MissingKey {
                path: path.to_path_buf(),
                key: "general.architecture",
            });
        }
    };
    if architecture != "llama" {
        return Err(StartupError::Architecture {
            path: path.to_path_buf(),
            found: architecture,
        });
    }

    let eos_id = match content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(meta_u32)
    {
        Some(id) => id,
        None => {
            return Err(StartupError::NoEos {
                path: path.to_path_buf(),
            });
        }
    };
    let vocab_size = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(tokens)) if !tokens.is_empty() => {
            u32::try_from(tokens.len()).map_err(|_| StartupError::NoVocabulary {
                path: path.to_path_buf(),
            })?
        }
        _ => {
            return Err(StartupError::NoVocabulary {
                path: path.to_path_buf(),
            });
        }
    };

    let context_length = require_u32(&content, path, "llama.context_length")?;
    let block_count = require_u32(&content, path, "llama.block_count")?;
    let head_count = require_u32(&content, path, "llama.attention.head_count")?;
    let head_count_kv = require_u32(&content, path, "llama.attention.head_count_kv")?;
    let embedding_length = require_u32(&content, path, "llama.embedding_length")?;
    let head_dim = head_dim(&content, path, head_count, embedding_length)?;
    let rope_freq_base = match content.metadata.get("llama.rope.freq_base") {
        Some(Value::F32(value)) => *value,
        Some(Value::F64(value)) => *value as f32,
        None => 10_000.0,
        Some(_) => {
            return Err(StartupError::MissingKey {
                path: path.to_path_buf(),
                key: "llama.rope.freq_base",
            });
        }
    };
    let add_bos = matches!(
        content.metadata.get("tokenizer.ggml.add_bos_token"),
        Some(Value::Bool(true))
    );
    let bos_id = if add_bos {
        require_u32(&content, path, "tokenizer.ggml.bos_token_id")?
    } else {
        0
    };
    let name = match content.metadata.get("general.name") {
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        _ => path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("model")
            .to_string(),
    };

    Ok(ModelCard {
        name,
        context_length,
        working_context: context_length.min(max_context),
        block_count,
        head_count,
        head_count_kv,
        embedding_length,
        head_dim,
        rope_freq_base,
        eos_id,
        add_bos,
        bos_id,
        vocab_size,
    })
}

fn classify_gguf_error(path: &Path, error: &candle_core::Error) -> StartupError {
    let text = error.to_string().to_ascii_lowercase();
    if text.contains("dtype") || text.contains("quant") {
        StartupError::UnsupportedQuantization {
            path: path.to_path_buf(),
        }
    } else {
        StartupError::Read {
            path: path.to_path_buf(),
        }
    }
}

fn meta_u32(value: &Value) -> Option<u32> {
    match value {
        Value::U32(value) => Some(*value),
        Value::U64(value) => u32::try_from(*value).ok(),
        Value::I32(value) if *value >= 0 => u32::try_from(*value).ok(),
        Value::I64(value) if *value >= 0 => u32::try_from(*value).ok(),
        _ => None,
    }
}

fn require_u32(content: &Content, path: &Path, key: &'static str) -> Result<u32, StartupError> {
    content
        .metadata
        .get(key)
        .and_then(meta_u32)
        .ok_or(StartupError::MissingKey {
            path: path.to_path_buf(),
            key,
        })
}

fn head_dim(
    content: &Content,
    path: &Path,
    head_count: u32,
    embedding_length: u32,
) -> Result<u32, StartupError> {
    if let Some(value) = content
        .metadata
        .get("llama.attention.key_length")
        .and_then(meta_u32)
    {
        return Ok(value);
    }
    if let Some(value) = content
        .metadata
        .get("llama.rope.dimension_count")
        .and_then(meta_u32)
    {
        return Ok(value);
    }
    if head_count > 0 && embedding_length.is_multiple_of(head_count) {
        Ok(embedding_length / head_count)
    } else {
        Err(StartupError::MissingKey {
            path: path.to_path_buf(),
            key: "llama.attention.key_length",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{StartupError, inspect_gguf};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn write(name: &str, bytes: &[u8]) -> Self {
            let tick = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("ingle-{name}-{tick}.gguf"));
            fs::write(&path, bytes).unwrap();
            Self { path }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn push_str(buf: &mut Vec<u8>, text: &str) {
        buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
        buf.extend_from_slice(text.as_bytes());
    }

    fn kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        push_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_str(buf, value);
    }

    fn kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        push_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        push_str(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
        push_str(buf, key);
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(u8::from(value));
    }

    fn kv_strings(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
        push_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            push_str(buf, value);
        }
    }

    fn gguf(tensor_count: u64, body: &[u8], kv_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tensor_count.to_le_bytes());
        bytes.extend_from_slice(&kv_count.to_le_bytes());
        bytes.extend_from_slice(body);
        bytes
    }

    fn llama_body(name: Option<&str>, rope: Option<f32>, key_length: Option<u32>, add_bos: bool) -> (Vec<u8>, u64) {
        let mut body = Vec::new();
        let mut count = 0u64;
        let mut add = |write: &dyn Fn(&mut Vec<u8>)| {
            write(&mut body);
            count += 1;
        };
        add(&|buf| kv_string(buf, "general.architecture", "llama"));
        if let Some(name) = name {
            add(&|buf| kv_string(buf, "general.name", name));
        }
        add(&|buf| kv_u32(buf, "llama.context_length", 8192));
        add(&|buf| kv_u32(buf, "llama.block_count", 22));
        add(&|buf| kv_u32(buf, "llama.attention.head_count", 32));
        add(&|buf| kv_u32(buf, "llama.attention.head_count_kv", 4));
        add(&|buf| kv_u32(buf, "llama.embedding_length", 2048));
        if let Some(rope) = rope {
            add(&|buf| kv_f32(buf, "llama.rope.freq_base", rope));
        }
        if let Some(key_length) = key_length {
            add(&|buf| kv_u32(buf, "llama.attention.key_length", key_length));
        }
        add(&|buf| kv_u32(buf, "tokenizer.ggml.eos_token_id", 2));
        add(&|buf| kv_strings(buf, "tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>"]));
        if add_bos {
            add(&|buf| kv_bool(buf, "tokenizer.ggml.add_bos_token", true));
            add(&|buf| kv_u32(buf, "tokenizer.ggml.bos_token_id", 1));
        }
        (body, count)
    }

    fn error_text(path: &Path) -> String {
        inspect_gguf(path, 4096).unwrap_err().to_string()
    }

    #[test]
    fn a_missing_path_names_the_file() {
        let path = std::env::temp_dir().join("ingle-missing-model.gguf");
        let _ = fs::remove_file(&path);
        let error = inspect_gguf(&path, 4096).unwrap_err();
        assert_eq!(error, StartupError::FileNotFound { path: path.clone() });
        assert_eq!(error.to_string(), format!("file not found: {}", path.display()));
    }

    #[test]
    fn a_bad_magic_is_not_a_gguf() {
        let file = TempFile::write("bad-magic", b"not a model");
        let error = inspect_gguf(&file.path, 4096).unwrap_err();
        assert_eq!(error, StartupError::NotGguf { path: file.path.clone() });
        assert!(error.to_string().contains("not a GGUF"));
    }

    #[test]
    fn a_truncated_gguf_is_a_read_error() {
        let file = TempFile::write("truncated", b"GGUF");
        assert_eq!(
            inspect_gguf(&file.path, 4096).unwrap_err(),
            StartupError::Read { path: file.path.clone() }
        );
    }

    #[test]
    fn an_unknown_dtype_is_an_unsupported_quantization() {
        let mut body = Vec::new();
        push_str(&mut body, "blk.0.attn_q.weight");
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&1u64.to_le_bytes());
        body.extend_from_slice(&999u32.to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes());
        let file = TempFile::write("bad-dtype", &gguf(1, &body, 0));
        assert_eq!(
            inspect_gguf(&file.path, 4096).unwrap_err(),
            StartupError::UnsupportedQuantization { path: file.path.clone() }
        );
    }

    #[test]
    fn another_architecture_names_the_value_found() {
        let mut body = Vec::new();
        kv_string(&mut body, "general.architecture", "qwen2");
        let file = TempFile::write("qwen", &gguf(0, &body, 1));
        let text = error_text(&file.path);
        assert!(text.contains("qwen2"), "{text}");
        assert!(text.contains("llama"), "{text}");
        assert!(text.contains(&file.path.display().to_string()), "{text}");
    }

    #[test]
    fn a_missing_eos_id_is_its_own_error() {
        let mut body = Vec::new();
        kv_string(&mut body, "general.architecture", "llama");
        let file = TempFile::write("no-eos", &gguf(0, &body, 1));
        let text = error_text(&file.path);
        assert!(text.contains("tokenizer.ggml.eos_token_id"), "{text}");
        assert!(!text.contains("qwen2"));
    }

    #[test]
    fn an_empty_vocabulary_is_rejected() {
        let mut body = Vec::new();
        kv_string(&mut body, "general.architecture", "llama");
        kv_u32(&mut body, "tokenizer.ggml.eos_token_id", 2);
        kv_strings(&mut body, "tokenizer.ggml.tokens", &[]);
        let file = TempFile::write("empty-vocab", &gguf(0, &body, 3));
        assert_eq!(
            inspect_gguf(&file.path, 4096).unwrap_err(),
            StartupError::NoVocabulary { path: file.path.clone() }
        );
    }

    #[test]
    fn the_card_comes_from_metadata_and_the_working_context_is_capped() {
        let (body, count) = llama_body(Some("tiny"), Some(500_000.0), Some(64), true);
        let file = TempFile::write("llama", &gguf(0, &body, count));
        let card = inspect_gguf(&file.path, 4096).unwrap();
        assert_eq!(card.name, "tiny");
        assert_eq!(card.context_length, 8192);
        assert_eq!(card.working_context, 4096);
        assert_eq!(card.block_count, 22);
        assert_eq!(card.head_count, 32);
        assert_eq!(card.head_count_kv, 4);
        assert_eq!(card.embedding_length, 2048);
        assert_eq!(card.head_dim, 64);
        assert_eq!(card.rope_freq_base, 500_000.0);
        assert_eq!(card.eos_id, 2);
        assert!(card.add_bos);
        assert_eq!(card.bos_id, 1);
        assert_eq!(card.vocab_size, 3);
    }

    #[test]
    fn the_file_stem_names_the_model_when_general_name_is_absent() {
        let (body, count) = llama_body(None, None, None, false);
        let file = TempFile::write("stem-name", &gguf(0, &body, count));
        let card = inspect_gguf(&file.path, 4096).unwrap();
        assert!(card.name.starts_with("ingle-stem-name-"), "{}", card.name);
        assert_eq!(card.working_context, 4096);
        assert_eq!(card.head_dim, 2048 / 32);
        assert_eq!(card.rope_freq_base, 10_000.0);
        assert!(!card.add_bos);
    }

    #[test]
    fn a_shorter_model_context_is_left_unchanged() {
        let mut body = Vec::new();
        let mut count = 0u64;
        let mut add = |write: &dyn Fn(&mut Vec<u8>)| {
            write(&mut body);
            count += 1;
        };
        add(&|buf| kv_string(buf, "general.architecture", "llama"));
        add(&|buf| kv_u32(buf, "llama.context_length", 1024));
        add(&|buf| kv_u32(buf, "llama.block_count", 1));
        add(&|buf| kv_u32(buf, "llama.attention.head_count", 1));
        add(&|buf| kv_u32(buf, "llama.attention.head_count_kv", 1));
        add(&|buf| kv_u32(buf, "llama.embedding_length", 8));
        add(&|buf| kv_u32(buf, "tokenizer.ggml.eos_token_id", 0));
        add(&|buf| kv_strings(buf, "tokenizer.ggml.tokens", &["a"]));
        let file = TempFile::write("short-ctx", &gguf(0, &body, count));
        let card = inspect_gguf(&file.path, 4096).unwrap();
        assert_eq!(card.context_length, 1024);
        assert_eq!(card.working_context, 1024);
    }
}
