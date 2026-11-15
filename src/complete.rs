use std::sync::atomic::{AtomicBool, Ordering};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::chat::{AcceptedChat, chatml_prompt, render_template};
use crate::encode::{decode_tokens, encode_prompt};
use crate::generate::{Model, ModelError, TokenRow};
use crate::sample::{SampleParams, greedy_token, sample_token};
use crate::stop::StopHold;

/// Weights and vocabulary the completion loop reads.
pub struct Loaded {
    pub vocab: Vec<String>,
    pub eos_id: u32,
    pub add_bos: bool,
    pub bos_id: u32,
    pub working_context: u32,
    pub template: Option<String>,
}

/// Text the client can append, in order.
#[derive(Debug, PartialEq, Eq)]
pub struct Completion {
    pub pieces: Vec<String>,
    pub finish_reason: &'static str,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Completion {
    pub fn text(&self) -> String {
        self.pieces.concat()
    }
}

/// The request did not become a completion.
#[derive(Debug, PartialEq, Eq)]
pub enum CompleteError {
    ContextLength,
    Tokenize,
    Template,
    Generate,
    Cancelled,
}

struct Prepared {
    ids: Vec<u32>,
    prompt_tokens: u32,
}

/// Render and encode the prompt. This does not run the model.
/// Render the prompt and reject one that does not fit the working context.
pub(crate) fn fits(loaded: &Loaded, chat: &AcceptedChat) -> Result<(), CompleteError> {
    prepare(loaded, chat).map(|_| ())
}

fn prepare(loaded: &Loaded, chat: &AcceptedChat) -> Result<Prepared, CompleteError> {
    let prompt = match &loaded.template {
        Some(template) => {
            render_template(template, &chat.messages).map_err(|_| CompleteError::Template)?
        }
        None => chatml_prompt(&chat.messages),
    };
    let ids = encode_prompt(&prompt, &loaded.vocab, loaded.add_bos, loaded.bos_id)
        .map_err(|_| CompleteError::Tokenize)?;
    let prompt_tokens = u32::try_from(ids.len()).map_err(|_| CompleteError::Generate)?;
    if prompt_tokens > loaded.working_context {
        return Err(CompleteError::ContextLength);
    }
    Ok(Prepared { ids, prompt_tokens })
}

/// Score `chat` on `model`. `cancelled` is checked between tokens.
pub fn complete<M: Model + ?Sized>(
    model: &mut M,
    loaded: &Loaded,
    chat: &AcceptedChat,
    cancelled: &AtomicBool,
) -> Result<Completion, CompleteError> {
    let prepared = prepare(loaded, chat)?;
    score(model, loaded, chat, &prepared, cancelled)
}

fn score<M: Model + ?Sized>(
    model: &mut M,
    loaded: &Loaded,
    chat: &AcceptedChat,
    prepared: &Prepared,
    cancelled: &AtomicBool,
) -> Result<Completion, CompleteError> {
    let prompt_ids = &prepared.ids;
    let prompt_tokens = prepared.prompt_tokens;
    let room = loaded.working_context - prompt_tokens;
    let max_new = chat.max_tokens.min(room);
    if max_new == 0 {
        return Ok(Completion {
            pieces: Vec::new(),
            finish_reason: "length",
            prompt_tokens,
            completion_tokens: 0,
        });
    }

    let mut logits = last_row(model.forward(&rows(prompt_ids)?))?;
    let mut rng = ChaCha20Rng::seed_from_u64(chat.seed.unwrap_or_else(os_seed));
    let mut generated = Vec::new();
    let mut pieces = Vec::new();
    let mut hold = StopHold::default();
    let mut position = prompt_tokens;

    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err(CompleteError::Cancelled);
        }
        if generated.len() as u32 == max_new {
            push_piece(&mut pieces, hold.flush());
            return Ok(done(pieces, "length", prompt_tokens, generated.len()));
        }
        let id = choose(chat, &logits, &generated, &mut rng)?;
        if id == loaded.eos_id {
            push_piece(&mut pieces, hold.flush());
            return Ok(done(pieces, "stop", prompt_tokens, generated.len()));
        }
        let delta = token_delta(&loaded.vocab, &generated, id)?;
        generated.push(id);
        let (visible, stopped) = hold.push(&delta, &chat.stop);
        push_piece(&mut pieces, visible);
        if stopped {
            return Ok(done(pieces, "stop", prompt_tokens, generated.len()));
        }
        if generated.len() as u32 == max_new {
            push_piece(&mut pieces, hold.flush());
            return Ok(done(pieces, "length", prompt_tokens, generated.len()));
        }
        logits = last_row(model.forward(&[row(id, position)?]))?;
        position += 1;
    }
}

fn os_seed() -> u64 {
    rand::rng().random()
}

fn choose(
    chat: &AcceptedChat,
    logits: &[f32],
    generated: &[u32],
    rng: &mut ChaCha20Rng,
) -> Result<u32, CompleteError> {
    let id = if chat.temperature == 0.0 {
        greedy_token(logits)
    } else {
        sample_token(
            logits,
            generated,
            rng,
            &SampleParams {
                temperature: chat.temperature,
                top_p: chat.top_p,
                top_k: chat.top_k,
                frequency_penalty: chat.frequency_penalty,
                presence_penalty: chat.presence_penalty,
            },
        )
    };
    id.ok_or(CompleteError::Generate)
}

fn token_delta(vocab: &[String], generated: &[u32], id: u32) -> Result<String, CompleteError> {
    let mut ids = generated.to_vec();
    ids.push(id);
    let previous = decode_tokens(generated, vocab).map_err(|_| CompleteError::Generate)?;
    let next = decode_tokens(&ids, vocab).map_err(|_| CompleteError::Generate)?;
    if let Some(delta) = next.strip_prefix(&previous) {
        Ok(delta.to_string())
    } else {
        Ok(next)
    }
}

fn done(
    pieces: Vec<String>,
    finish_reason: &'static str,
    prompt_tokens: u32,
    completion_tokens: usize,
) -> Completion {
    Completion {
        pieces,
        finish_reason,
        prompt_tokens,
        completion_tokens: u32::try_from(completion_tokens).unwrap_or(u32::MAX),
    }
}

fn push_piece(pieces: &mut Vec<String>, piece: String) {
    if !piece.is_empty() {
        pieces.push(piece);
    }
}

fn rows(ids: &[u32]) -> Result<Vec<TokenRow>, CompleteError> {
    ids.iter()
        .enumerate()
        .map(|(position, id)| {
            row(
                *id,
                u32::try_from(position).map_err(|_| CompleteError::Generate)?,
            )
        })
        .collect()
}

fn row(token_id: u32, position: u32) -> Result<TokenRow, CompleteError> {
    Ok(TokenRow { token_id, position })
}

fn last_row(rows: Result<Vec<Vec<f32>>, ModelError>) -> Result<Vec<f32>, CompleteError> {
    rows.map_err(|_| CompleteError::Generate)?
        .into_iter()
        .next_back()
        .ok_or(CompleteError::Generate)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::{CompleteError, Loaded, complete};
    use crate::chat::accept_chat;
    use crate::generate::{Model, ModelError, TokenRow};
    use serde_json::json;

    struct Script {
        rows: Vec<Vec<f32>>,
        cursor: usize,
    }

    impl Model for Script {
        fn forward(&mut self, batch: &[TokenRow]) -> Result<Vec<Vec<f32>>, ModelError> {
            let row = self.rows.get(self.cursor).cloned().ok_or(ModelError)?;
            self.cursor += 1;
            Ok(vec![row; batch.len()])
        }
    }

    fn vocab() -> Vec<String> {
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
        .map(str::to_string)
        .to_vec()
    }

    fn loaded(context: u32) -> Loaded {
        Loaded {
            vocab: vocab(),
            eos_id: 7,
            add_bos: false,
            bos_id: 0,
            working_context: context,
            template: None,
        }
    }

    fn logits(winner: usize) -> Vec<f32> {
        let mut row = vec![0.0; 8];
        row[winner] = 1.0;
        row
    }

    #[test]
    fn a_greedy_reply_omits_eos_and_counts_tokens() {
        let chat = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":8}),
            "tiny",
        )
        .unwrap();
        let mut model = Script {
            rows: vec![logits(6), logits(7)],
            cursor: 0,
        };
        let completion =
            complete(&mut model, &loaded(4096), &chat, &AtomicBool::new(false)).unwrap();
        assert_eq!(completion.text(), "yo");
        assert_eq!(completion.finish_reason, "stop");
        assert_eq!(completion.completion_tokens, 1);
        assert_eq!(completion.prompt_tokens, 9);
        assert_eq!(model.cursor, 2);
    }

    #[test]
    fn the_context_cap_stops_with_length() {
        let chat = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":8}),
            "tiny",
        )
        .unwrap();
        let mut model = Script {
            rows: vec![logits(6), logits(6), logits(6)],
            cursor: 0,
        };
        let completion = complete(&mut model, &loaded(10), &chat, &AtomicBool::new(false)).unwrap();
        assert_eq!(completion.text(), "yo");
        assert_eq!(completion.finish_reason, "length");
        assert_eq!(completion.completion_tokens, 1);
    }

    #[test]
    fn a_prompt_past_the_context_is_rejected() {
        let chat = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}]}),
            "tiny",
        )
        .unwrap();
        let mut model = Script {
            rows: Vec::new(),
            cursor: 0,
        };
        let error = complete(&mut model, &loaded(1), &chat, &AtomicBool::new(false)).unwrap_err();
        assert_eq!(error, CompleteError::ContextLength);
        assert_eq!(model.cursor, 0);
    }

    #[test]
    fn a_stop_string_is_not_delivered() {
        let chat = accept_chat(
            &json!({"model":"tiny","messages":[{"role":"user","content":"hi"}],"temperature":0,"stop":"yo"}),
            "tiny",
        )
        .unwrap();
        let mut model = Script {
            rows: vec![logits(6), logits(6)],
            cursor: 0,
        };
        let completion =
            complete(&mut model, &loaded(4096), &chat, &AtomicBool::new(false)).unwrap();
        assert_eq!(completion.text(), "");
        assert_eq!(completion.finish_reason, "stop");
        assert_eq!(completion.completion_tokens, 1);
    }
}
