/// Index of the greatest logit. Equal values resolve to the smaller index.
pub fn greedy_token(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (index, logit) in logits.iter().copied().enumerate() {
        let id = u32::try_from(index).ok()?;
        match best {
            None => best = Some((id, logit)),
            Some((_, best_logit)) if logit > best_logit => best = Some((id, logit)),
            Some(_) => {}
        }
    }
    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::greedy_token;

    #[test]
    fn picks_the_greater_logit() {
        assert_eq!(greedy_token(&[0.1, 0.4, 0.2]), Some(1));
    }

    #[test]
    fn tie_keeps_the_smaller_id() {
        assert_eq!(greedy_token(&[1.0, 0.5, 1.0]), Some(0));
    }

    #[test]
    fn later_tie_keeps_the_smaller_id() {
        assert_eq!(greedy_token(&[0.2, 3.0, 3.0]), Some(1));
    }

    #[test]
    fn empty_logits_have_no_token() {
        assert_eq!(greedy_token(&[]), None);
    }
}
