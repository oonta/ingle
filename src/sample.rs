use std::collections::HashMap;

use rand::Rng;
use rand_chacha::ChaCha20Rng;

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

/// Knobs applied when temperature is greater than zero. Zero temperature stays greedy.
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

/// One draw after penalties, temperature, top-k, and top-p. An empty nucleus falls back to argmax.
pub fn sample_token(
    logits: &[f32],
    generated: &[u32],
    rng: &mut ChaCha20Rng,
    params: &SampleParams,
) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if params.temperature == 0.0 {
        return greedy_token(logits);
    }
    let mut scores = logits.to_vec();
    apply_penalties(
        &mut scores,
        generated,
        params.frequency_penalty,
        params.presence_penalty,
    );
    for score in &mut scores {
        *score /= params.temperature;
    }
    if let Some(k) = params.top_k {
        mask_top_k(&mut scores, k);
    }
    mask_top_p(&mut scores, params.top_p);
    draw(&scores, rng).or_else(|| greedy_token(logits))
}

fn apply_penalties(scores: &mut [f32], generated: &[u32], frequency: f32, presence: f32) {
    let mut counts = HashMap::<u32, u32>::new();
    for id in generated {
        *counts.entry(*id).or_insert(0) += 1;
    }
    for (id, count) in counts {
        let Some(score) = usize::try_from(id)
            .ok()
            .and_then(|index| scores.get_mut(index))
        else {
            continue;
        };
        *score -= count as f32 * frequency;
        if count > 0 {
            *score -= presence;
        }
    }
}

fn mask_top_k(scores: &mut [f32], k: u32) {
    let keep = usize::try_from(k).unwrap_or(usize::MAX).min(scores.len());
    if keep == 0 || keep >= scores.len() {
        return;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });
    let cutoff = scores[order[keep - 1]];
    let mut kept = 0usize;
    for index in order {
        if kept < keep && scores[index] >= cutoff {
            kept += 1;
        } else if kept >= keep {
            scores[index] = f32::NEG_INFINITY;
        }
    }
}

fn mask_top_p(scores: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let probabilities = softmax(scores);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    let mut sum = 0.0;
    let mut keep = vec![false; scores.len()];
    for index in order {
        if sum >= top_p && keep.iter().any(|kept| *kept) {
            break;
        }
        keep[index] = true;
        sum += probabilities[index];
    }
    if !keep.iter().any(|kept| *kept) {
        return;
    }
    for (score, kept) in scores.iter_mut().zip(keep) {
        if !kept {
            *score = f32::NEG_INFINITY;
        }
    }
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents: Vec<f32> = scores
        .iter()
        .map(|score| {
            if score.is_finite() {
                (score - max).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = exponents.iter().sum();
    if sum == 0.0 {
        return vec![0.0; scores.len()];
    }
    exponents.into_iter().map(|value| value / sum).collect()
}

fn draw(scores: &[f32], rng: &mut ChaCha20Rng) -> Option<u32> {
    let probabilities = softmax(scores);
    let sum: f32 = probabilities.iter().sum();
    if sum == 0.0 {
        return None;
    }
    let mut pick = rng.random::<f32>() * sum;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if pick <= probability {
            return u32::try_from(index).ok();
        }
        pick -= probability;
    }
    u32::try_from(probabilities.len().saturating_sub(1)).ok()
}

#[cfg(test)]
mod tests {
    use super::{SampleParams, greedy_token, sample_token};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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

    #[test]
    fn temperature_zero_stays_greedy() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        assert_eq!(
            sample_token(
                &[0.1, 0.4, 0.2],
                &[],
                &mut rng,
                &params(0.0, 0.2, Some(1), 2.0, 2.0),
            ),
            Some(1)
        );
    }

    #[test]
    fn the_same_seed_draws_the_same_token() {
        let once = draw(7);
        assert_eq!(once, draw(7));
        assert_eq!(once, 0);
    }

    #[test]
    fn a_repeated_token_is_penalized_below_its_rival() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let id = sample_token(
            &[10.0, 0.0],
            &[0],
            &mut rng,
            &params(1.0, 1.0, None, 100.0, 0.0),
        );
        assert_eq!(id, Some(1));
    }

    fn draw(seed: u64) -> u32 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        sample_token(
            &[0.0, 0.2, 0.1],
            &[],
            &mut rng,
            &params(1.0, 1.0, None, 0.0, 0.0),
        )
        .unwrap()
    }

    fn params(
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> SampleParams {
        SampleParams {
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
        }
    }
}
