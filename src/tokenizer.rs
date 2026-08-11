use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::{
    Tokenizer,
    tokenizer::{
        DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
        PreTokenizerWrapper,
    },
};

pub type QwenDecodeStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

pub struct QwenTokenizer {
    inner: Tokenizer,
}

impl QwenTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u64>> {
        let encoding = self
            .inner
            .encode(prompt, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to tokenize prompt")?;
        Ok(encoding.get_ids().iter().map(|&id| u64::from(id)).collect())
    }

    pub fn decode(&self, token_ids: &[u64], skip_special_tokens: bool) -> Result<String> {
        let ids = token_ids
            .iter()
            .map(|&id| u32::try_from(id).context("token ID exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode tokens")
    }

    pub fn decode_stream(&self, skip_special_tokens: bool) -> QwenDecodeStream<'_> {
        self.inner.decode_stream(skip_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downloaded_tokenizer_round_trips_when_present() -> Result<()> {
        let path = Path::new("models/Qwen3-30B-A3B-Base/tokenizer.json");
        if path.exists() {
            let tokenizer = QwenTokenizer::open(path)?;
            let tokens = tokenizer.encode("Gravity is")?;
            assert!(!tokens.is_empty());
            assert_eq!(tokenizer.decode(&tokens, false)?, "Gravity is");
        }
        Ok(())
    }

    #[test]
    fn downloaded_tokenizer_streams_when_present() -> Result<()> {
        let path = Path::new("models/Qwen3-30B-A3B-Base/tokenizer.json");
        if path.exists() {
            let tokenizer = QwenTokenizer::open(path)?;
            let tokens = tokenizer.encode("Gravity is the only thing")?;
            let mut stream = tokenizer.decode_stream(false);
            let mut decoded = String::new();
            for token in tokens {
                if let Some(chunk) = stream
                    .step(u32::try_from(token)?)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?
                {
                    decoded.push_str(&chunk);
                }
            }
            assert_eq!(decoded, "Gravity is the only thing");
        }
        Ok(())
    }
}
