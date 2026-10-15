//! Cooperative kernels for the experimental backend, implemented entirely in Hex.
use super::*;

pub(crate) fn source(config: &ModelConfig, options: InferenceOptions) -> Result<String> {
    let mut text = super::graph_source(
        config,
        options.with_experimental_kernels(ExperimentalKernelKind::Reference),
    )?;
    // Keep the reference primitives available for attention and normalization.
    // Redirect projection call sites, preserving the public artifact ABI.
    text = text
        .replace("} qwen3.project [", "} qwen3.fused.project [")
        .replace("} qwen3.normalize [", "} qwen3.fused.normalize [");
    let start = text.find("(def program qwen3.decoder-layer").unwrap();
    let decoder_end = text[start..].find("\n ))").unwrap() + start + 5;
    let mut decoder = text[start..decoder_end].to_owned();
    for (input, weight, features, residual, target, tail) in [
        (
            "context-ready",
            "o-weight",
            config.query_size(),
            "hidden",
            "attention-result",
            "  ({[.attention-result] [.post-norm]",
        ),
        (
            "gated",
            "down-weight",
            config.intermediate_size,
            "attention-result",
            "result-final",
            "  ({[.cache-ready] [.result-final]",
        ),
    ] {
        let begin = decoder
            .find(&format!("  ({{[.{input}] [.{weight}]"))
            .unwrap();
        let end = decoder[begin..].find(tail).unwrap() + begin;
        let mut fused = String::new();
        call(
            &mut fused,
            &[
                input,
                weight,
                &c(features),
                &c(config.hidden_size),
                residual,
            ],
            "qwen3.fused.residual",
            "fused-result",
        );
        call(
            &mut fused,
            &[input, "fused-result"],
            "qwen3.free-bf16-before",
            "freed-input",
        );
        call(
            &mut fused,
            &[residual, "freed-input"],
            "qwen3.free-bf16-before",
            "freed-residual",
        );
        if target == "attention-result" {
            call(
                &mut fused,
                &["normalized", "freed-residual"],
                "qwen3.free-bf16-before",
                target,
            );
        } else {
            fused.push_str(&format!("  ([.freed-residual] [{target}.])\n"));
        }
        // Distinct local names for the two residual fusions.
        if target == "result-final" {
            fused = fused
                .replace("fused-result", "ffn-fused-result")
                .replace("freed-input", "ffn-freed-input")
                .replace("freed-residual", "ffn-freed-residual");
        }
        decoder.replace_range(begin..end, &fused);
    }
    let begin = decoder.find("  ({[.ffn-input] [.gate-weight]").unwrap();
    let end = decoder[begin..]
        .find("  ({[.gated] [.down-weight]")
        .unwrap()
        + begin;
    let mut gate = String::new();
    call(
        &mut gate,
        &[
            "ffn-input",
            "gate-weight",
            &c(config.hidden_size),
            &c(config.intermediate_size),
            "up-weight",
        ],
        "qwen3.fused.gate",
        "activated",
    );
    call(
        &mut gate,
        &["ffn-input", "activated"],
        "qwen3.free-bf16-before",
        "gated",
    );
    decoder.replace_range(begin..end, &gate);
    text.replace_range(start..decoder_end, &decoder);
    append_hex(
        &mut text,
        include_str!("catena/experimental/cooperative.hex"),
    );
    text.push_str(include_str!("catena/experimental/fused_projection.hex"));
    text.push_str(
        &include_str!("catena/experimental/fused_normalization.hex").replace(
            "{{RMS_EPSILON_BITS}}",
            &format!("{:08X}", config.rms_norm_eps.to_bits()),
        ),
    );
    text.push_str(include_str!("catena/experimental/fused_argmax.hex"));
    text = text.replace("} qwen3.argmax [", "} qwen3.fused.argmax [");
    let entry = text.find("(def program qwen3 :").unwrap();
    let entry_end = balanced_end(&text, entry);
    let mut graph = text[entry..entry_end].to_owned();
    graph = graph.replace(
        &format!("qwen3 : {{{REF} {OWN}"),
        &format!("qwen3 : {{{REF} {OWN} {OWN}"),
    );
    graph = graph.replace(
        &format!("-> {{{OWN} {OWN} {OWN}}}"),
        &format!("-> {{{OWN} {OWN} {OWN} {OWN}}}"),
    );
    graph = graph.replace(
        "tokens cache capacity start",
        "tokens cache workspace capacity start",
    );
    graph = graph.replace("{[.logits-final] } qwen3.fused.argmax [next-token.]", "{[.workspace] [.logits-final]} qwen3.fused.argmax-with-workspace [sampled.])\n  ([.sampled] *.elim [workspace-result next-token.]");
    graph = graph.replace(
        "{[.cache-final] [.logits-final] [.next-token]}",
        "{[.cache-final] [.workspace-result] [.logits-final] [.next-token]}",
    );
    text.replace_range(entry..entry_end, &graph);
    Ok(text)
}

fn append_hex(text: &mut String, source: &str) {
    let mut source = source.to_owned();
    for declaration in ["(theory type nat {", "(theory program type {"] {
        if let Some(start) = source.find(declaration) {
            let end = balanced_end(&source, start);
            let body = source[start + declaration.len()..end - 2].to_owned();
            let target = text
                .find(declaration)
                .expect("reference metatype declarations");
            let target_end = balanced_end(text, target);
            text.insert_str(target_end - 2, &body);
            source.replace_range(start..end, "");
        }
    }
    text.push_str(&source);
}
fn balanced_end(text: &str, start: usize) -> usize {
    let mut depth = 0;
    for (offset, ch) in text[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return start + offset + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unterminated Hex declaration")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_projection_compiles() -> Result<()> {
        let options =
            InferenceOptions::new(InferencePrecision::Bf16).with_codegen(CodegenKind::Experimental);
        let source = source(&super::super::tests::tiny_config(), options)?;
        let module = compile(
            crate::catena::runtime_sources_for_options(&source, options),
            GpuDialect::Hip,
        )?;
        let entry = module
            .functions
            .iter()
            .find(|f| f.source_name == "qwen3")
            .unwrap();
        use catena_lang::runtime::ValueKind;
        assert_eq!(
            entry.inputs[..5],
            [
                ValueKind::MemRef,
                ValueKind::MemOwn,
                ValueKind::MemOwn,
                ValueKind::U64,
                ValueKind::U64
            ]
        );
        assert_eq!(entry.outputs, vec![ValueKind::MemOwn; 4]);
        assert!(module.source.contains("__syncthreads"));
        Ok(())
    }
}
