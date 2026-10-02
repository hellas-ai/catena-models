use super::Config;
use anyhow::Result;
pub const SOURCE: &str = include_str!("../catena/tts_model.hex");
/// Complete portable Hex bundle; the receiver supplies only Catena's stdlib.
pub fn bundle(config: &Config) -> Result<String> {
    config.validate()?;
    Ok([
        crate::catena::BF16_HELPER_SOURCE,
        crate::whisper::graph::HELPERS,
        include_str!("../catena/inference_helpers.hex"),
        SOURCE,
    ]
    .join("\n"))
}
