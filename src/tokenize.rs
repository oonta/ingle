/// Prepend `bos_id` once when the flag is set and the sequence does not already start with it.
pub fn with_bos(tokens: &[u32], add_bos: bool, bos_id: u32) -> Vec<u32> {
    if add_bos && tokens.first() != Some(&bos_id) {
        let mut with_prefix = Vec::with_capacity(tokens.len() + 1);
        with_prefix.push(bos_id);
        with_prefix.extend_from_slice(tokens);
        with_prefix
    } else {
        tokens.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::with_bos;

    #[test]
    fn flag_off_leaves_tokens_unchanged() {
        assert_eq!(with_bos(&[2, 3], false, 1), vec![2, 3]);
    }

    #[test]
    fn flag_on_prepends_when_missing() {
        assert_eq!(with_bos(&[2, 3], true, 1), vec![1, 2, 3]);
    }

    #[test]
    fn flag_on_does_not_prepend_twice() {
        assert_eq!(with_bos(&[1, 2], true, 1), vec![1, 2]);
    }

    #[test]
    fn empty_prompt_gains_a_bos() {
        assert_eq!(with_bos(&[], true, 1), vec![1]);
    }
}
