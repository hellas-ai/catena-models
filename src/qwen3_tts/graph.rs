//! Checkpoint bindings and the non-streaming talker/predictor HEX graph.
use super::{Binding, Config, Sampling, Storage, codec, kernels::*, prefill};
use crate::{
    catena::{self, InferenceOptions},
    qwen::{self, c, call, header, nested_product, pack, unpack},
    spec::{ModelConfig, WeightDtype, weight_specs},
};
use anyhow::Result;
use std::{collections::BTreeMap, fmt::Write};

const PAIR: &str = "({((cap.own mem) val) ((cap.own mem) val)} *)";
const PREFIX: &str = "stdlib.qwen-tts.";
const BATCH_PROJECT: &str = "stdlib.qwen35.batch.wmma-project";

pub(super) fn build(
    config: &Config,
    sampling: &Sampling,
) -> Result<(String, Vec<Binding>, Vec<Binding>)> {
    let talker = &config.talker_config;
    let primary = talker.decoder.model_config();
    let predictor = talker.code_predictor_config.model_config();
    let mut s = qwen::graph_source(&primary, InferenceOptions::default())?;
    // Keep Qwen3's shared cached layer kernels and drop the text LM entry points.
    let first_entry = s.find("(def program stdlib.qwen3.token-step").unwrap();
    let first_extra = s.find("(def program stdlib.models.coop.").unwrap();
    s.replace_range(first_entry..first_extra, "");
    let predictor_source = qwen::graph_source(&predictor, InferenceOptions::default())?;
    let predictor_programs = catena::definitions(
        &predictor_source,
        &[
            "qwen3.decoder-layer:",
            "stdlib.qwen3.select-layer",
            "stdlib.qwen3.layer-weights:",
            "stdlib.qwen3.decoder-step.fold:",
        ],
    );
    s.push_str(
        &predictor_programs
            .replace("qwen3.decoder-layer", "qwen-tts.predictor.layer")
            .replace(
                "stdlib.qwen3.select-layer",
                "stdlib.qwen-tts.predictor.select-layer",
            )
            .replace(
                "stdlib.qwen3.layer-weights",
                "stdlib.qwen-tts.predictor.layer-weights",
            )
            .replace(
                "stdlib.qwen3.decoder-step.fold",
                "stdlib.qwen-tts.predictor.layer-step",
            ),
    );
    for fragment in [
        include_str!("../catena/qwen35_batch_wmma.hex"),
        include_str!("../catena/qwen35_batch_projection.hex"),
        include_str!("../catena/qwen35_prefill.hex"),
        include_str!("../catena/bf16_simt.hex"),
        include_str!("../catena/bf16_attention.hex"),
        include_str!("../catena/whisper.hex"),
        include_str!("../catena/state_buffers.hex"),
        include_str!("../catena/tts_model.hex"),
    ] {
        qwen::fused::append_hex(&mut s, fragment);
    }
    s.push_str("(def program stdlib.qwen-tts.conv-project : {((cap.own mem) val) ((cap.ref mem) val) (u64 val) (u64 val) (u64 val)} -> ((cap.own mem) val) = ([x w rows ci co.] ({[.x] [.w] [.ci] [.co]} stdlib.qwen35.batch.wmma-project)))\n");
    s = s.replace(
        "} stdlib.whisper.linear [t7.]",
        "} stdlib.qwen-tts.conv-project [t7.]",
    );
    s.push_str(include_str!("../catena/qwen3_tts_qkv.hex"));
    fuse_qkv(&mut s, "qwen3.decoder-layer", &primary);
    fuse_qkv(&mut s, "qwen-tts.predictor.layer", &predictor);
    emit_generation(&mut s);
    let primary_weights = decoder(&mut s, &primary, "talker", "stdlib.qwen3.decoder-step.fold");
    let predictor_weights = decoder(
        &mut s,
        &predictor,
        "predictor",
        "stdlib.qwen-tts.predictor.layer-step",
    );
    let mut model = Vec::new();
    model.extend(decoder_bindings(&primary, "talker")?);
    model.extend(decoder_bindings(&predictor, "talker.code_predictor")?);
    let primary_hidden = primary.hidden_size;
    let text_hidden = talker.text_hidden_size;
    for (name, shape) in [
        (
            "talker.model.text_embedding.weight",
            vec![talker.text_vocab_size, text_hidden],
        ),
        (
            "talker.model.codec_embedding.weight",
            vec![primary.vocabulary_size, primary_hidden],
        ),
        (
            "talker.codec_head.weight",
            vec![primary.vocabulary_size, primary_hidden],
        ),
        (
            "talker.text_projection.linear_fc1.weight",
            vec![text_hidden, text_hidden],
        ),
        ("talker.text_projection.linear_fc1.bias", vec![text_hidden]),
        (
            "talker.text_projection.linear_fc2.weight",
            vec![primary_hidden, text_hidden],
        ),
        (
            "talker.text_projection.linear_fc2.bias",
            vec![primary_hidden],
        ),
    ] {
        model.push(Binding::new(name, shape, Storage::Bf16));
    }
    let projection = primary_hidden != predictor.hidden_size;
    if projection {
        model.push(Binding::new(
            "talker.code_predictor.small_to_mtp_projection.weight",
            [predictor.hidden_size, primary_hidden],
            Storage::Bf16,
        ));
        model.push(Binding::new(
            "talker.code_predictor.small_to_mtp_projection.bias",
            [predictor.hidden_size],
            Storage::Bf16,
        ));
    }
    for code in 0..15 {
        model.push(Binding::new(
            format!("talker.code_predictor.model.codec_embedding.{code}.weight"),
            [2048, primary_hidden],
            Storage::Bf16,
        ));
        model.push(Binding::new(
            format!("talker.code_predictor.lm_head.{code}.weight"),
            [2048, predictor.hidden_size],
            Storage::Bf16,
        ));
    }
    let (codec_program, codec_bindings) = codec::emit();
    s.push_str(&codec_program);
    conditioning(&mut s, config);
    prefill::emit(&mut s, &primary, &primary_weights);
    let book_type = book_selector(&mut s);
    frame(
        &mut s,
        config,
        sampling,
        &primary_weights,
        &predictor_weights,
        &book_type,
    );
    entry(&mut s, config, &model, codec_bindings.len());
    s = super::guard::generation(&s);
    let single_call = super::schedule::supported();
    let aliases: BTreeMap<_, _> = s
        .split("(def program ")
        .skip(1)
        .filter_map(|definition| {
            let name = definition.split_whitespace().next()?;
            (!name.starts_with("stdlib.")
                && (!name.starts_with("qwen3-tts.")
                    || (single_call && name != "qwen3-tts.synthesize")))
                .then(|| (name.to_owned(), format!("stdlib.qwen-tts.shared.{name}")))
        })
        .collect();
    Ok((catena::rename_programs(&s, &aliases), model, codec_bindings))
}

fn decoder_bindings(config: &ModelConfig, prefix: &str) -> Result<Vec<Binding>> {
    let specs = weight_specs(config)?;
    let mut bindings = specs
        .iter()
        .skip(1)
        .take(config.num_layers * 11)
        .map(|spec| {
            Binding::new(
                format!("{prefix}.{}", spec.name),
                spec.shape.clone(),
                if spec.dtype == WeightDtype::F32 {
                    Storage::F32
                } else {
                    Storage::Bf16
                },
            )
        })
        .collect::<Vec<_>>();
    bindings.push(Binding::new(
        format!("{prefix}.model.norm.weight"),
        [config.hidden_size],
        Storage::F32,
    ));
    Ok(bindings)
}

fn decoder(s: &mut String, config: &ModelConfig, name: &str, layer: &str) -> String {
    let group = nested_product(&vec![REF.to_owned(); 11]);
    let weights = nested_product(&vec![group; config.num_layers]);
    header(
        s,
        &format!("{PREFIX}{name}.token"),
        &[
            ("weights", &weights),
            ("capacity", U64),
            ("position", U64),
            ("cache", OWN),
            ("hidden", OWN),
        ],
        PAIR,
    );
    call(s, &["position"], "qwen3.rope-one", "rope");
    call(s, &["cache", "hidden"], "*.intro", "initial");
    fold(
        s,
        layer,
        &["weights", "rope", "capacity", "position"],
        &u(config.num_layers),
        "initial",
        "decoded",
    );
    s.push_str("  ([.decoded] *.elim [next-cache next-hidden.])\n");
    free(s, &["rope"], "next-hidden", "result", FREE_F32);
    finish(s, "({[.next-cache] [.result]} *.intro)");
    weights
}

fn conditioning(s: &mut String, config: &Config) {
    let args = [
        ("text", REF),
        ("text-len", U64),
        ("instruction", REF),
        ("instruction-len", U64),
        ("codec", REF),
        ("codec-len", U64),
        ("text-emb", REF),
        ("codec-emb", REF),
        ("fc1", REF),
        ("bias1", REF),
        ("fc2", REF),
        ("bias2", REF),
    ];
    let text_width = c(config.talker_config.text_hidden_size);
    let width = c(config.talker_config.decoder.hidden_size);
    header(s, &format!("{PREFIX}conditioning"), &args, PAIR);
    mem(
        s,
        "u64.add",
        &["text-len", "instruction-len"],
        "text-instruct",
    );
    mem(s, "u64.add", &["text-instruct", "codec-len"], "combined");
    mem(s, "u64.add", &["combined", &c(3)], "rows"); // includes trailing text-pad row
    mem(s, "u64.mul", &["rows", &text_width], "text-elements");
    kernel(
        s,
        "prompt-text",
        &[
            "text",
            "text-len",
            "instruction",
            "instruction-len",
            "codec-len",
            &c(config.tts_pad_token_id as usize),
            &c(config.tts_bos_token_id as usize),
            &c(config.tts_eos_token_id as usize),
            "text-emb",
            &text_width,
            "text-elements",
        ],
        "embedded",
    );
    call(
        s,
        &["embedded", "fc1", &text_width, &text_width],
        BATCH_PROJECT,
        "projected1",
    );
    kernel(
        s,
        "bias-silu",
        &["projected1", "bias1", &text_width, "text-elements"],
        "activated-raw",
    );
    free(
        s,
        &["embedded", "projected1"],
        "activated-raw",
        "activated",
        FREE_BF16,
    );
    call(
        s,
        &["activated", "fc2", &text_width, &width],
        BATCH_PROJECT,
        "projected2",
    );
    mem(s, "u64.mul", &["rows", &width], "elements");
    kernel(
        s,
        "bias",
        &["projected2", "bias2", &width, "elements"],
        "text-projected",
    );
    mem(s, "u64.sub", &["rows", &c(1)], "prompt-rows");
    mem(s, "u64.mul", &["prompt-rows", &width], "prompt-elements");
    kernel(
        s,
        "prompt-codec",
        &[
            "text-projected",
            "instruction-len",
            "codec",
            "codec-len",
            "codec-emb",
            &c(config.talker_config.codec_pad_id as usize),
            &c(config.talker_config.codec_bos_id as usize),
            "prompt-rows",
            &width,
            "prompt-elements",
        ],
        "conditioned",
    );
    kernel(
        s,
        "row",
        &["text-projected", "prompt-rows", &width, &width],
        "pad-row",
    );
    free(
        s,
        &["activated", "projected2", "text-projected"],
        "pad-row",
        "pad",
        FREE_BF16,
    );
    finish(s, "({[.conditioned] [.pad]} *.intro)");
}

fn book_selector(s: &mut String) -> String {
    let group = nested_product(&[REF.to_owned(), REF.to_owned()]);
    let books = nested_product(&vec![group.clone(); 15]);
    let mut selector = String::new();
    qwen::emit_layer_selector(&mut selector, 15, &group, &books);
    s.push_str(
        &selector
            .replace("stdlib.qwen3.select-layer", "stdlib.qwen-tts.select-book")
            .replace("stdlib.qwen3.layer-weights", "stdlib.qwen-tts.book-weights"),
    );
    books
}

fn norm(s: &mut String, input: &str, weights: &str, width: usize, out: &str) {
    call(
        s,
        &[input, weights, &c(width), &f(width as f32), &c(1)],
        "qwen3.fused.normalize",
        out,
    );
}

fn project(s: &mut String, input: &str, weights: &str, width: usize, outputs: usize, out: &str) {
    call(
        s,
        &[input, weights, &c(width), &c(outputs)],
        "qwen3.fused.project",
        out,
    );
}

fn frame(
    s: &mut String,
    cfg: &Config,
    sampling: &Sampling,
    talker_weights: &str,
    predictor_weights: &str,
    books: &str,
) {
    let d = &cfg.talker_config;
    let h = d.decoder.hidden_size;
    let p = d.code_predictor_config.hidden_size;
    let codes = c(2048);
    let project_input = h != p;
    let count4 = nested_product(&vec![OWN.to_owned(); 4]);
    let count6 = nested_product(&vec![OWN.to_owned(); 6]);
    let mut predictor_args = vec![
        ("weights", predictor_weights),
        ("norm", REF),
        ("books", books),
        ("frame", U64),
        ("seed", U64),
        ("progress", OWN),
        ("codes", OWN),
    ];
    if project_input {
        predictor_args.extend([("mtp", REF), ("mtp-bias", REF)]);
        header(
            s,
            &format!("{PREFIX}predictor.input"),
            &[("input", OWN), ("mtp", REF), ("mtp-bias", REF)],
            OWN,
        );
        project(s, "input", "mtp", h, p, "projected");
        kernel(
            s,
            "bias",
            &["projected", "mtp-bias", &c(p), &c(p)],
            "biased",
        );
        free(s, &["projected"], "biased", "result", FREE_BF16);
        finish(s, "[.result]");
    }

    for (name, last) in [("book", false), ("last-book", true)] {
        step_header(
            s,
            &format!("{PREFIX}predictor.{name}"),
            &predictor_args,
            count4.as_str(),
        );
        unpack(
            s,
            "state",
            &["cache", "hidden", "sum", "next-codes"].map(str::to_owned),
        );
        let index = if last {
            mem(s, "u64.add", &["i", &c(14)], "last-index");
            "last-index"
        } else {
            "i"
        };
        mem(s, "fold-count-u32", &[index], "book-index");
        call(
            s,
            &["books", "book-index"],
            "stdlib.qwen-tts.book-weights",
            "selected",
        );
        s.push_str("  ([.selected] *.elim [embedding head.])\n");
        norm(s, "hidden", "norm", p, "normalized");
        project(s, "normalized", "head", p, 2048, "logits");
        mem(s, "u64.add", &[index, &c(1)], "group");
        kernel(
            s,
            "sample",
            &[
                "logits",
                "progress",
                "codes",
                "frame",
                "group",
                "seed",
                &codes,
                &c(sampling.subtalker_top_k),
                &f(sampling.subtalker_temperature),
                &f(1.0),
                &c(1),
            ],
            "sampled",
        );
        mem(s, "u64.mul", &["frame", &c(16)], "frame-offset");
        mem(s, "u64.add", &["frame-offset", "group"], "code-offset");
        kernel(
            s,
            "store-code.into",
            &["sampled", "code-offset", "next-codes", &c(1)],
            "written",
        );
        kernel(
            s,
            "embed-code",
            &["sampled", "embedding", &c(h), &c(h)],
            "embedded",
        );
        kernel(
            s,
            "sum-embeddings.into",
            &["embedded", "sum", &c(h)],
            "summed",
        );
        if last {
            free(
                s,
                &["embedded", "normalized", "logits"],
                "summed",
                "last-sum",
                FREE_BF16,
            );
            free(s, &["sampled"], "last-sum", "result-sum", FREE_F32);
            pack(
                s,
                &["cache", "hidden", "result-sum", "written"].map(str::to_owned),
                "result",
            );
            step_finish(s, "result");
            continue;
        }
        if project_input {
            kernel(
                s,
                "predictor.input",
                &["embedded", "mtp", "mtp-bias"],
                "predictor-row",
            );
            free(s, &["embedded"], "predictor-row", "row", FREE_BF16);
        } else {
            s.push_str("  ([.embedded] [row.])\n");
        }
        free(
            s,
            &["hidden", "normalized", "logits"],
            "row",
            "input",
            FREE_BF16,
        );
        mem(s, "free", &["sampled"], "released");
        s.push_str("  ([.released] unit.elim)\n");
        mem(s, "u64.add", &["i", &c(2)], "position");
        kernel(
            s,
            "predictor.token",
            &["weights", &c(17), "position", "cache", "input"],
            "decoded",
        );
        s.push_str("  ([.decoded] *.elim [next-cache next-hidden.])\n");
        pack(
            s,
            &["next-cache", "next-hidden", "summed", "written"].map(str::to_owned),
            "result",
        );
        step_finish(s, "result");
    }

    let mut frame_args = vec![
        ("weights", talker_weights),
        ("norm", REF),
        ("predictor-weights", predictor_weights),
        ("predictor-norm", REF),
        ("books", books),
        ("codec-embedding", REF),
        ("codec-head", REF),
        ("capacity", U64),
        ("prompt-rows", U64),
        ("seed", U64),
        ("pad", OWN),
        ("start", U64),
    ];
    if project_input {
        frame_args.extend([("mtp", REF), ("mtp-bias", REF)]);
    }
    step_header(s, &format!("{PREFIX}frame"), &frame_args, &count6);
    mem(s, "u64.add", &["start", "i"], "frame-index");
    unpack(
        s,
        "state",
        &[
            "cache",
            "hidden",
            "predictor-cache",
            "sum",
            "codes",
            "progress",
        ]
        .map(str::to_owned),
    );
    norm(s, "hidden", "norm", h, "normalized");
    project(
        s,
        "normalized",
        "codec-head",
        h,
        d.decoder.vocab_size,
        "logits",
    );
    kernel(
        s,
        "sample",
        &[
            "logits",
            "progress",
            "codes",
            "frame-index",
            &c(0),
            "seed",
            &c(d.codec_eos_token_id as usize),
            &c(sampling.top_k),
            &f(sampling.temperature),
            &f(sampling.repetition_penalty),
            &c(1),
        ],
        "sampled",
    );
    kernel(
        s,
        "progress.into",
        &[
            "sampled",
            "frame-index",
            &c(d.codec_eos_token_id as usize),
            "progress",
            &c(1),
        ],
        "updated",
    );
    mem(s, "u64.mul", &["frame-index", &c(16)], "offset");
    kernel(
        s,
        "store-code.into",
        &["sampled", "offset", "codes", &c(1)],
        "first-code",
    );
    kernel(
        s,
        "embed-code",
        &["sampled", "codec-embedding", &c(h), &c(h)],
        "first-embedding",
    );
    kernel(
        s,
        "init-sum.into",
        &["first-embedding", "sum", &c(h)],
        "sum-first",
    );
    if project_input {
        kernel(
            s,
            "predictor.input",
            &["normalized", "mtp", "mtp-bias"],
            "prev",
        );
        kernel(
            s,
            "predictor.input",
            &["first-embedding", "mtp", "mtp-bias"],
            "first-row",
        );
        free(s, &["first-embedding"], "first-row", "first", FREE_BF16);
    } else {
        kernel(s, "row", &["normalized", &c(0), &c(h), &c(h)], "prev");
        s.push_str("  ([.first-embedding] [first.])\n");
    }
    free(
        s,
        &["hidden", "normalized", "logits"],
        "prev",
        "context",
        FREE_BF16,
    );
    mem(s, "free", &["sampled"], "released");
    s.push_str("  ([.released] unit.elim)\n");
    kernel(
        s,
        "predictor.token",
        &[
            "predictor-weights",
            &c(17),
            &c(0),
            "predictor-cache",
            "context",
        ],
        "first-predicted",
    );
    s.push_str("  ([.first-predicted] *.elim [context-cache context-hidden.])\n");
    free(
        s,
        &["context-hidden"],
        "first",
        "predictor-input",
        FREE_BF16,
    );
    kernel(
        s,
        "predictor.token",
        &[
            "predictor-weights",
            &c(17),
            &c(1),
            "context-cache",
            "predictor-input",
        ],
        "predictor-start",
    );
    s.push_str("  ([.predictor-start] *.elim [book-cache book-hidden.])\n");
    pack(
        s,
        &["book-cache", "book-hidden", "sum-first", "first-code"].map(str::to_owned),
        "book-init",
    );
    let mut captures = vec![
        "predictor-weights",
        "predictor-norm",
        "books",
        "frame-index",
        "seed",
        "updated",
        "first-code",
    ];
    if project_input {
        captures.extend(["mtp", "mtp-bias"]);
    }
    fold(
        s,
        "stdlib.qwen-tts.predictor.book",
        &captures,
        &u(14),
        "book-init",
        "books-prefix",
    );
    fold(
        s,
        "stdlib.qwen-tts.predictor.last-book",
        &captures,
        &u(1),
        "books-prefix",
        "books-final",
    );
    unpack(
        s,
        "books-final",
        &[
            "next-predictor-cache",
            "last-book-hidden",
            "frame-sum",
            "next-codes",
        ]
        .map(str::to_owned),
    );
    kernel(s, "sum-pad", &["frame-sum", "pad", &c(h)], "next-input");
    free(
        s,
        &["last-book-hidden"],
        "next-input",
        "next-row",
        FREE_BF16,
    );
    mem(s, "u64.add", &["prompt-rows", "frame-index"], "position");
    kernel(
        s,
        "talker.token",
        &["weights", "capacity", "position", "cache", "next-row"],
        "decoded",
    );
    s.push_str("  ([.decoded] *.elim [next-cache next-hidden.])\n");
    pack(
        s,
        &[
            "next-cache",
            "next-hidden",
            "next-predictor-cache",
            "frame-sum",
            "next-codes",
            "updated",
        ]
        .map(str::to_owned),
        "result",
    );
    step_finish(s, "result");
}

fn entry(s: &mut String, cfg: &Config, model: &[Binding], codec_len: usize) {
    let names = (0..model.len() + codec_len)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>();
    let mut inputs = vec![
        ("text", REF),
        ("text-len", U64),
        ("instruction", REF),
        ("instruction-len", U64),
        ("codec", REF),
        ("codec-len", U64),
        ("max-frames", U32),
    ];
    inputs.extend(names[..model.len()].iter().map(|n| (n.as_str(), REF)));
    let state = [
        "prompt-cache",
        "ready-hidden",
        "predictor-cache",
        "sum",
        "codes",
        "progress",
        "pad",
    ];
    let state_types = vec![OWN.to_owned(); state.len()];
    let outputs = format!("{{{}}}", state_types.join(" "));
    header(s, "qwen3-tts.prepare", &inputs, &outputs);
    let talker = &cfg.talker_config;
    let primary = &talker.decoder;
    let predictor = &talker.code_predictor_config;
    let offset = primary.num_hidden_layers * 11 + predictor.num_hidden_layers * 11 + 2;
    pack_decoder(s, &names, "talker", primary.num_hidden_layers);
    let mut cond = vec![
        "text",
        "text-len",
        "instruction",
        "instruction-len",
        "codec",
        "codec-len",
    ];
    cond.extend(
        [
            offset,
            offset + 1,
            offset + 3,
            offset + 4,
            offset + 5,
            offset + 6,
        ]
        .map(|i| names[i].as_str()),
    );
    kernel(s, "conditioning", &cond, "conditioning");
    s.push_str("  ([.conditioning] *.elim [prompt pad.])\n");
    mem(s, "u32.to-u64", &["max-frames"], "budget");
    mem(s, "own-count-bf16", &["prompt"], "prompt-elements");
    mem(
        s,
        "u64.div",
        &["prompt-elements", &c(primary.hidden_size)],
        "prompt-rows",
    );
    mem(s, "u64.add", &["prompt-rows", "budget"], "capacity");
    mem(
        s,
        "u64.mul",
        &[
            "capacity",
            &c(primary.num_hidden_layers * 2 * primary.num_key_value_heads * primary.head_dim),
        ],
        "cache-count",
    );
    mem(s, "alloc-bf16", &["cache-count"], "cache");
    kernel(s, "zero-u64", &[&c(2)], "progress");
    kernel(
        s,
        "prefill",
        &[
            "talker-weights",
            "prompt-rows",
            "capacity",
            "cache",
            "prompt",
        ],
        "prefilled",
    );
    s.push_str("  ([.prefilled] *.elim [prompt-cache ready-hidden.])\n");
    mem(
        s,
        "alloc-bf16",
        &[&c(17
            * predictor.num_hidden_layers
            * 2
            * predictor.num_key_value_heads
            * predictor.head_dim)],
        "predictor-cache",
    );
    mem(s, "alloc-f32", &[&c(primary.hidden_size)], "sum");
    mem(s, "u64.mul", &["budget", &c(16)], "code-count");
    kernel(s, "zero-u64", &["code-count"], "codes");
    finish(
        s,
        "{[.prompt-cache] [.ready-hidden] [.predictor-cache] [.sum] [.codes] [.progress] [.pad]}",
    );

    let mut step = vec![
        ("start", U64),
        ("steps", U32),
        ("capacity", U64),
        ("prompt-rows", U64),
        ("seed", U64),
    ];
    step.extend(state.iter().map(|name| (*name, OWN)));
    step.extend(names[..model.len()].iter().map(|n| (n.as_str(), REF)));
    header(s, "qwen3-tts.advance", &step, &outputs);
    pack_decoder(s, &names, "talker", primary.num_hidden_layers);
    pack_decoder(
        s,
        &names[primary.num_hidden_layers * 11 + 1..],
        "predictor",
        predictor.num_hidden_layers,
    );
    let extra = if primary.hidden_size != predictor.hidden_size {
        2
    } else {
        0
    };
    let first_book = offset + 7 + extra;
    let groups = (0..15)
        .map(|i| {
            let name = format!("book{i}");
            pack(s, &names[first_book + 2 * i..first_book + 2 * i + 2], &name);
            name
        })
        .collect::<Vec<_>>();
    pack(s, &groups, "books");
    pack(
        s,
        &state[..6]
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
        "frame-initial",
    );
    let mut captures = vec![
        "talker-weights",
        "talker-norm",
        "predictor-weights",
        "predictor-norm",
        "books",
        &names[offset + 1],
        &names[offset + 2],
        "capacity",
        "prompt-rows",
        "seed",
        "pad",
    ];
    captures.push("start");
    captures.extend(names[offset + 7..first_book].iter().map(String::as_str));
    fold(
        s,
        "stdlib.qwen-tts.frame",
        &captures,
        "[.steps]",
        "frame-initial",
        "generated",
    );
    unpack(
        s,
        "generated",
        &[
            "final-cache",
            "last-hidden",
            "final-predictor-cache",
            "final-sum",
            "generated-codes",
            "descriptor",
        ]
        .map(str::to_owned),
    );
    finish(
        s,
        "{[.final-cache] [.last-hidden] [.final-predictor-cache] [.final-sum] [.generated-codes] [.descriptor] [.pad]}",
    );

    let mut decode = vec![("codes", OWN), ("progress", OWN), ("frames", U64)];
    decode.extend(names[model.len()..].iter().map(|name| (name.as_str(), REF)));
    header(s, "qwen3-tts.decode", &decode, OWN);
    let args = decode.iter().map(|(name, _)| *name).collect::<Vec<_>>();
    kernel(s, "decode", &args, "wave");
    free(s, &["codes", "progress"], "wave", "result", FREE_F32);
    finish(s, "[.result]");
    if super::schedule::supported() {
        super::schedule::emit(s, &inputs, &names, model.len(), &state);
    }
}

fn pack_decoder(s: &mut String, names: &[String], name: &str, layers: usize) {
    let groups = (0..layers)
        .map(|layer| {
            let label = format!("{name}-layer-{layer}");
            pack(s, &names[layer * 11..(layer + 1) * 11], &label);
            label
        })
        .collect::<Vec<_>>();
    pack(s, &groups, &format!("{name}-weights"));
    writeln!(s, "  ([.{}] [{name}-norm.])", names[layers * 11]).unwrap();
}

fn fuse_qkv(s: &mut String, name: &str, config: &ModelConfig) {
    let layer = s.find(&format!("(def program {name} :")).unwrap();
    let begin = layer + s[layer..].find("  ({[.normalized] [.q-weight]").unwrap();
    let end = begin + s[begin..].find("  ({[.query-raw] [.q-norm]").unwrap();
    assert!(config.query_size().is_multiple_of(4) && config.key_value_size().is_multiple_of(4));
    let mut fused = String::new();
    call(
        &mut fused,
        &[
            "progress",
            "normalized",
            "q-weight",
            "k-weight",
            "v-weight",
            &c(config.hidden_size),
            &c(config.query_size()),
            &c(config.key_value_size()),
        ],
        "stdlib.qwen-tts.qkv",
        "qkv",
    );
    unpack(
        &mut fused,
        "qkv",
        &["query-raw", "key-raw", "value"].map(str::to_owned),
    );
    s.replace_range(begin..end, &fused);
}
