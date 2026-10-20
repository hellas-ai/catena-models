use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use tokenizers::{
    AddedToken, Tokenizer,
    models::bpe::{BPE, Vocab},
    normalizers::unicode::NFC,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence as PreSequence,
        split::{Split, SplitPattern},
    },
    processors::{
        PostProcessorWrapper, sequence::Sequence as PostSequence, template::TemplateProcessing,
    },
    tokenizer::{
        DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, SplitDelimiterBehavior,
    },
};

use crate::checkpoint::{GgufTokenizerMetadata, read_gguf_tokenizer_metadata};

const QWEN2_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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

    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Option<Self>> {
        read_gguf_tokenizer_metadata(path)?
            .map(Self::from_gguf_metadata)
            .transpose()
    }

    fn from_gguf_metadata(metadata: GgufTokenizerMetadata) -> Result<Self> {
        ensure!(
            metadata.model.as_deref() == Some("gpt2"),
            "unsupported GGUF tokenizer model {:?}; expected gpt2",
            metadata.model
        );
        let split_pattern = match metadata.pre.as_deref() {
            Some("qwen2") => QWEN2_SPLIT_PATTERN,
            Some("qwen35") => QWEN35_SPLIT_PATTERN,
            other => bail!("unsupported GGUF pre-tokenizer {other:?}; expected qwen2 or qwen35"),
        };
        ensure!(
            metadata.tokens.len() <= u32::MAX as usize,
            "GGUF tokenizer vocabulary exceeds u32"
        );
        if !metadata.token_types.is_empty() {
            ensure!(
                metadata.token_types.len() == metadata.tokens.len(),
                "GGUF tokenizer has {} tokens but {} token types",
                metadata.tokens.len(),
                metadata.token_types.len()
            );
        }

        let mut vocab = Vocab::with_capacity(metadata.tokens.len());
        for (id, token) in metadata.tokens.iter().enumerate() {
            ensure!(
                vocab.insert(token.clone(), id as u32).is_none(),
                "duplicate GGUF tokenizer token {token:?}"
            );
        }
        let merges = metadata
            .merges
            .iter()
            .map(|merge| {
                let (left, right) = merge
                    .split_once(' ')
                    .with_context(|| format!("invalid GGUF BPE merge {merge:?}"))?;
                ensure!(
                    !left.is_empty() && !right.is_empty() && !right.contains(' '),
                    "invalid GGUF BPE merge {merge:?}"
                );
                Ok((left.to_owned(), right.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to build GGUF BPE model")?;

        let byte_level = ByteLevel::new(false, false, false);
        let split = Split::new(
            SplitPattern::Regex(split_pattern.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to build Qwen pre-tokenizer regex")?;
        let mut inner = Tokenizer::new(bpe);
        inner.with_normalizer(Some(NFC));
        inner.with_pre_tokenizer(Some(PreSequence::new(vec![
            PreTokenizerWrapper::from(split),
            PreTokenizerWrapper::from(byte_level),
        ])));
        inner.with_decoder(Some(byte_level));

        let template = special_token_template(&metadata)?;
        if let Some(template) = template {
            inner.with_post_processor(Some(PostSequence::new(vec![
                PostProcessorWrapper::from(byte_level),
                PostProcessorWrapper::from(template),
            ])));
        } else {
            inner.with_post_processor(Some(byte_level));
        }

        let mut special_ids = metadata
            .special_token_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        for (id, &token_type) in metadata.token_types.iter().enumerate() {
            if token_type == 3 {
                special_ids.insert(id as u32);
            }
        }
        let special_tokens = special_ids
            .into_iter()
            .map(|id| {
                metadata
                    .tokens
                    .get(id as usize)
                    .map(|token| AddedToken::from(token.clone(), true))
                    .with_context(|| format!("GGUF special token ID {id} is out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        inner.add_special_tokens(&special_tokens);

        let user_tokens = metadata
            .token_types
            .iter()
            .enumerate()
            .filter(|(_, token_type)| **token_type == 4)
            .map(|(id, _)| AddedToken::from(metadata.tokens[id].clone(), false).normalized(false))
            .collect::<Vec<_>>();
        inner.add_tokens(&user_tokens);
        Ok(Self { inner })
    }

    pub fn special_token_id(&self, token: &str) -> Result<u64> {
        self.inner
            .token_to_id(token)
            .map(u64::from)
            .with_context(|| format!("tokenizer is missing {token}"))
    }

    pub fn encode_without_special_tokens(&self, prompt: &str) -> Result<Vec<u64>> {
        let encoded = self
            .inner
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(encoded.get_ids().iter().map(|&id| u64::from(id)).collect())
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

fn special_token_template(metadata: &GgufTokenizerMetadata) -> Result<Option<TemplateProcessing>> {
    if !metadata.add_bos_token && !metadata.add_eos_token {
        return Ok(None);
    }
    let mut pieces = Vec::new();
    let mut special_tokens = Vec::new();
    if metadata.add_bos_token {
        let id = metadata
            .bos_token_id
            .context("GGUF tokenizer enables BOS but has no BOS token ID")?;
        let token = metadata
            .tokens
            .get(id as usize)
            .with_context(|| format!("GGUF BOS token ID {id} is out of range"))?;
        pieces.push(token.as_str());
        special_tokens.push((token.clone(), id));
    }
    pieces.push("$A");
    if metadata.add_eos_token {
        let id = metadata
            .eos_token_id
            .context("GGUF tokenizer enables EOS but has no EOS token ID")?;
        let token = metadata
            .tokens
            .get(id as usize)
            .with_context(|| format!("GGUF EOS token ID {id} is out of range"))?;
        pieces.push(token.as_str());
        if !special_tokens.iter().any(|(_, existing)| *existing == id) {
            special_tokens.push((token.clone(), id));
        }
    }
    TemplateProcessing::builder()
        .try_single(pieces.join(" "))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .special_tokens(special_tokens)
        .build()
        .map(Some)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to build GGUF special-token processor")
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer as PreTokenizerTrait,
    };

    use super::*;

    fn embedded_test_metadata(pre: &str, add_eos_token: bool) -> GgufTokenizerMetadata {
        GgufTokenizerMetadata {
            model: Some("gpt2".to_owned()),
            pre: Some(pre.to_owned()),
            tokens: ["h", "i", "hi", "<|end|>"].map(str::to_owned).to_vec(),
            merges: vec!["h i".to_owned()],
            token_types: vec![1, 1, 1, 3],
            special_token_ids: vec![3],
            eos_token_id: Some(3),
            add_eos_token,
            ..GgufTokenizerMetadata::default()
        }
    }

    #[test]
    fn builds_qwen_tokenizer_from_gguf_metadata() -> Result<()> {
        let tokenizer = QwenTokenizer::from_gguf_metadata(embedded_test_metadata("qwen2", false))?;
        assert_eq!(tokenizer.encode("hi")?, [2]);
        assert_eq!(tokenizer.decode(&[2], false)?, "hi");
        assert_eq!(tokenizer.encode("<|end|>")?, [3]);
        assert_eq!(tokenizer.decode(&[3], true)?, "");
        Ok(())
    }

    #[test]
    fn applies_gguf_eos_setting() -> Result<()> {
        let tokenizer = QwenTokenizer::from_gguf_metadata(embedded_test_metadata("qwen2", true))?;
        assert_eq!(tokenizer.encode("hi")?, [2, 3]);
        Ok(())
    }

    #[test]
    fn qwen35_keeps_combining_marks_with_letters() -> Result<()> {
        fn split(pattern: &str) -> Result<Vec<String>> {
            let tokenizer = Split::new(
                SplitPattern::Regex(pattern.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let mut text = PreTokenizedString::from("a\u{035c}b");
            tokenizer
                .pre_tokenize(&mut text)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            Ok(text
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(text, _, _)| text.to_owned())
                .collect())
        }

        assert_eq!(split(QWEN2_SPLIT_PATTERN)?, ["a", "\u{035c}b"]);
        assert_eq!(split(QWEN35_SPLIT_PATTERN)?, ["a\u{035c}b"]);
        Ok(())
    }
}
