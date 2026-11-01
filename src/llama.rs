use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::generate::{Model, ModelError, TokenRow};
use crate::load::{ModelCard, StartupError, card_from, classify_gguf_error, read_gguf};
use std::path::Path;

/// One resident llama model. Its cache is the single sequence `generate` scores.
pub struct LlamaModel {
    weights: ModelWeights,
    device: Device,
}

impl Model for LlamaModel {
    fn forward(&mut self, batch: &[TokenRow]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut rows = Vec::with_capacity(batch.len());
        for row in batch {
            let input = Tensor::new(&[row.token_id], &self.device)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|_| ModelError)?;
            let position = usize::try_from(row.position).map_err(|_| ModelError)?;
            let logits = self
                .weights
                .forward(&input, position)
                .map_err(|_| ModelError)?;
            rows.push(logit_row(logits).map_err(|_| ModelError)?);
        }
        Ok(rows)
    }
}

/// Load weights and the card from one file. `max_context` caps the working context.
pub fn load_llama(path: &Path, max_context: u32) -> Result<(ModelCard, LlamaModel), StartupError> {
    let (mut file, content) = read_gguf(path)?;
    let card = card_from(path, max_context, &content)?;
    let device = Device::Cpu;
    let weights = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|error| classify_gguf_error(path, &error))?;
    Ok((card, LlamaModel { weights, device }))
}

fn logit_row(mut logits: Tensor) -> candle_core::Result<Vec<f32>> {
    while logits.rank() > 1 {
        logits = logits.squeeze(0)?;
    }
    logits.to_vec1()
}
