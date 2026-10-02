use crate::{hfhub, tts::MAX_TOKENS};
use anyhow::{Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
/// MMS English uses characters, not the generic tokenizer.json format.
pub struct Tokenizer {
    vocab: BTreeMap<char, u64>,
}
impl Tokenizer {
    pub fn read(dir: &Path) -> Result<Self> {
        let config: serde_json::Value = hfhub::read_json(dir.join("tokenizer_config.json"))?;
        for (key, value) in [
            ("language", serde_json::json!("eng")),
            ("normalize", serde_json::json!(true)),
            ("add_blank", serde_json::json!(true)),
            ("phonemize", serde_json::json!(false)),
            ("is_uroman", serde_json::json!(false)),
            ("pad_token", serde_json::json!("k")),
            ("unk_token", serde_json::json!("<unk>")),
        ] {
            ensure!(
                config.get(key) == Some(&value),
                "unsupported MMS tokenizer {key}"
            );
        }
        let special: serde_json::Value = hfhub::read_json(dir.join("special_tokens_map.json"))?;
        ensure!(
            special.get("pad_token") == Some(&serde_json::json!("k"))
                && special.get("unk_token") == Some(&serde_json::json!("<unk>")),
            "unsupported MMS special tokens"
        );
        let vocab: BTreeMap<String, u64> = hfhub::read_json(dir.join("vocab.json"))?;
        let mut chars = BTreeMap::new();
        for (text, id) in vocab {
            let mut it = text.chars();
            let c = it
                .next()
                .ok_or_else(|| anyhow::anyhow!("empty vocabulary character"))?;
            ensure!(it.next().is_none(), "expected character vocabulary");
            chars.insert(c, id);
        }
        ensure!(
            chars.len() == 38
                && chars.values().copied().collect::<BTreeSet<_>>() == (0..38).collect()
                && chars.get(&'k') == Some(&0),
            "unsupported MMS English vocabulary"
        );
        Ok(Self { vocab: chars })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u64>> {
        let normalized: String = text
            .to_lowercase()
            .chars()
            .filter(|c| self.vocab.contains_key(c))
            .collect();
        let text = normalized.trim();
        ensure!(!text.is_empty(), "text is empty after MMS normalization");
        let length = text.chars().count() * 2 + 1;
        ensure!(
            length <= MAX_TOKENS,
            "text exceeds {MAX_TOKENS} tokens after blank insertion"
        );
        let mut ids = Vec::with_capacity(length);
        ids.push(0);
        for c in text.chars() {
            ids.push(self.vocab[&c]);
            ids.push(0);
        }
        Ok(ids)
    }
}
