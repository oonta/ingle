use std::collections::HashMap;

use crate::tokenize::with_bos;

/// The vocabulary cannot represent the prompt. `at` is a byte offset after spaces become ▁.
#[derive(Debug, PartialEq, Eq)]
pub struct EncodeError {
    pub at: usize,
}

/// Token ids for `text`. Spaces become ▁. The longest vocabulary piece wins.
/// A byte with no piece uses `<0xHH>` when that token exists. BOS is prepended once when requested.
pub fn encode_prompt(
    text: &str,
    tokens: &[String],
    add_bos: bool,
    bos_id: u32,
) -> Result<Vec<u32>, EncodeError> {
    let mut pieces: HashMap<&str, u32> = HashMap::new();
    let mut byte_ids = [None; 256];
    for (index, token) in tokens.iter().enumerate() {
        let Ok(id) = u32::try_from(index) else {
            break;
        };
        pieces.entry(token.as_str()).or_insert(id);
        if let Some(byte) = byte_token(token) {
            if byte_ids[usize::from(byte)].is_none() {
                byte_ids[usize::from(byte)] = Some(id);
            }
        }
    }

    let normalized = text.replace(' ', "▁");
    let mut ids = Vec::new();
    let mut rest = normalized.as_str();
    while !rest.is_empty() {
        if let Some((id, len)) = longest_piece(&pieces, rest) {
            ids.push(id);
            rest = &rest[len..];
            continue;
        }
        let byte = rest.as_bytes()[0];
        let Some(id) = byte_ids[usize::from(byte)] else {
            return Err(EncodeError {
                at: normalized.len() - rest.len(),
            });
        };
        ids.push(id);
        rest = &rest[1..];
    }
    Ok(with_bos(&ids, add_bos, bos_id))
}

fn longest_piece(pieces: &HashMap<&str, u32>, text: &str) -> Option<(u32, usize)> {
    let mut end = text.len();
    while end > 0 {
        if text.is_char_boundary(end) {
            if let Some(id) = pieces.get(&text[..end]).copied() {
                return Some((id, end));
            }
        }
        end -= 1;
    }
    None
}

fn byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::{EncodeError, encode_prompt};

    fn vocab(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|token| (*token).to_string()).collect()
    }

    #[test]
    fn the_longest_piece_wins() {
        let ids = encode_prompt("ab", &vocab(&["<unk>", "a", "ab", "b"]), false, 0).unwrap();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn a_space_becomes_the_sentencepiece_marker() {
        let ids = encode_prompt("a b", &vocab(&["a", "▁b", " "]), false, 0).unwrap();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn the_first_copy_of_a_piece_keeps_the_smaller_id() {
        let ids = encode_prompt("a", &vocab(&["a", "a"]), false, 0).unwrap();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn an_unlisted_byte_uses_its_byte_token() {
        let ids = encode_prompt("A", &vocab(&["<0x41>"]), false, 0).unwrap();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn a_byte_without_a_token_is_an_error() {
        let error = encode_prompt("A", &vocab(&["b"]), false, 0).unwrap_err();
        assert_eq!(error, EncodeError { at: 0 });
    }

    #[test]
    fn bos_is_prepended_once() {
        let ids = encode_prompt("ab", &vocab(&["<s>", "ab"]), true, 0).unwrap();
        assert_eq!(ids, vec![0, 1]);
        let again = encode_prompt("<s>ab", &vocab(&["<s>", "ab"]), true, 0).unwrap();
        assert_eq!(again, vec![0, 1]);
    }
}
