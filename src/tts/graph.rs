use super::Config;
use anyhow::Result;
pub const SOURCE: &str = include_str!("../catena/tts_model.hex");
/// Complete portable Hex bundle; the receiver supplies only Catena's core stdlib.
pub fn bundle(config: &Config) -> Result<String> {
    config.validate()?;
    Ok(with_exports(&["tts.synthesize", "tts.synthesize-debug"]))
}

fn with_exports(exports: &[&str]) -> String {
    let mut source = include_str!("../catena/memory.hex").to_owned();
    for fragment in [
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/cooperative.hex"),
        include_str!("../catena/state_buffers.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/bf16_simt.hex"),
        include_str!("../catena/bf16_attention.hex"),
        include_str!("../catena/whisper.hex"),
        SOURCE,
    ] {
        crate::qwen::fused::append_hex(&mut source, fragment);
    }
    let mapping = source
        .split("(def program ")
        .skip(1)
        .filter_map(|definition| {
            let name = definition.split_whitespace().next()?;
            (!name.starts_with("stdlib.") && !exports.contains(&name))
                .then(|| (name.to_owned(), format!("stdlib.{name}")))
        })
        .collect();
    crate::catena::rename_programs(&source, &mapping)
        .replace(" bf16x2.dot-f32 ", " stdlib.models.math.dot-pair-f32 ")
}

#[cfg(test)]
mod tests {
    #[test]
    fn bundle_uses_portable_reductions() {
        let source = super::with_exports(&["tts.synthesize", "tts.synthesize-debug"]);
        assert!(!source.contains("bf16x2.dot-f32"));
        assert!(!super::SOURCE.contains("bf16x2.dot-f32"));
        assert!(source.contains("stdlib.models.simt.or"));
        let raw =
            metacat::theory::RawTheorySet::from_texts(crate::catena::runtime_sources(&source))
                .unwrap();
        let prepared = crate::catena::prepare_sources(raw).unwrap();
        let program = prepared.theories.get(&"program".parse().unwrap()).unwrap();
        assert!(
            !program
                .arrows
                .keys()
                .any(|n| n.as_str() == "stdlib.models.math.dot-pair-f32")
        );
    }
}
