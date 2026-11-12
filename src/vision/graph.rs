//! Generate one linear-use Catena composition; Rust does not execute encoder layers.
use super::{VisionConfig, VisionKind};
use std::collections::{BTreeMap, HashMap};

fn c(n: usize) -> String {
    format!("(const.u64.0x{n:016X})")
}
fn f(x: f32) -> String {
    format!("(const.u32.0x{:08X} u32.bitcast-f32)", x.to_bits())
}
fn w(name: &str) -> String {
    format!("[.{name}]")
}
struct Op {
    name: String,
    args: Vec<String>,
    out: String,
}
struct Builder {
    ops: Vec<Op>,
    weights: HashMap<String, String>,
}
impl Builder {
    fn op(&mut self, name: &str, args: Vec<String>) -> String {
        let out = format!("m{}", self.ops.len());
        self.ops.push(Op {
            name: name.into(),
            args,
            out: out.clone(),
        });
        out
    }
    fn weight(&self, name: &str) -> String {
        w(&self.weights[name])
    }
    fn linear(&mut self, x: &str, prefix: &str, rows: &str, ci: usize, co: usize) -> String {
        let bias = format!("{prefix}.bias");
        let bias = self.weight(if self.weights.contains_key(&bias) {
            &bias
        } else {
            "__zero"
        });
        self.op(
            "stdlib.vision.linear-own",
            vec![
                w(x),
                self.weight(&format!("{prefix}.weight")),
                bias,
                w(rows),
                c(ci),
                c(co),
            ],
        )
    }
    fn norm(&mut self, x: &str, prefix: &str, rows: &str, cols: usize, eps: f32) -> String {
        self.op(
            "stdlib.vision.norm",
            vec![
                w(x),
                self.weight(&format!("{prefix}.weight")),
                self.weight(&format!("{prefix}.bias")),
                w(rows),
                c(cols),
                c(1),
                c(0),
                c(2),
                f(eps),
            ],
        )
    }
    fn add(&mut self, x: &str, y: &str) -> String {
        self.op("stdlib.vision.add", vec![w(x), w(y), w("hidden-count")])
    }
    fn render(self, result: &str, names: &[String], lfm3: bool) -> String {
        let mut last = BTreeMap::new();
        for (i, op) in self.ops.iter().enumerate() {
            last.insert(op.out.clone(), i);
            for arg in &op.args {
                if let Some(name) = arg.strip_prefix("[.").and_then(|s| s.strip_suffix(']'))
                    && last.contains_key(name)
                {
                    last.insert(name.into(), i);
                }
            }
        }
        let mut aliases = BTreeMap::new();
        let mut body = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            let args: Vec<_> = op
                .args
                .iter()
                .map(|a| {
                    if let Some(n) = a.strip_prefix("[.").and_then(|s| s.strip_suffix(']')) {
                        w(aliases.get(n).map(String::as_str).unwrap_or(n))
                    } else {
                        a.clone()
                    }
                })
                .collect();
            body += &format!(" ({{{}}} {} [{}.])\n", args.join(" "), op.name, op.out);
            let mut continuation = op.out.clone();
            for (name, &end) in &last {
                if end != i || name == &op.out || name == result {
                    continue;
                }
                let next = format!("free-{i}-{name}");
                body += &format!(
                    " ({{{} {}}} stdlib.models.resources.free-f32-before [{next}.])\n",
                    w(aliases.get(name).map(String::as_str).unwrap_or(name)),
                    w(&continuation)
                );
                continuation = next;
            }
            aliases.insert(op.out.clone(), continuation);
        }
        body += &w(aliases.get(result).map(String::as_str).unwrap_or(result));
        let mut types = vec!["((cap.ref mem) val)"; 5];
        if lfm3 {
            types.push("(u64 val)");
        }
        types.extend(vec!["((cap.ref mem) val)"; names.len()]);
        let types = types.join(" ");
        let width = if lfm3 { "width " } else { "" };
        let weight_args: Vec<_> = (0..names.len()).map(|i| format!("w{i}")).collect();
        format!(
            "\n(def program vision.encode : {{{types}}} -> ((cap.own mem) val)\n= ([patches positions indices coefficients frequencies {width}{}.]\n{body}))\n",
            weight_args.join(" ")
        )
    }
}
pub(crate) fn source(config: &VisionConfig, names: &[String]) -> String {
    if config.kind == VisionKind::Lfm3 {
        return lfm3_source(config, names);
    }
    let h = config.hidden_size;
    let mut b = Builder {
        ops: vec![],
        weights: names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), format!("w{i}")))
            .collect(),
    };
    // Shape scalars come from the patch buffer at runtime; they do not trigger recompilation.
    let mut prelude = format!(
        " ([.patches] stdlib.models.memory.ref-count-f32 [elements.])\n ({{[.elements] {}}} stdlib.models.memory.u64.div [rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.div [merged-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [hidden-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [ff-count.])\n",
        c(3 * config.patch_size * config.patch_size),
        c(config.merge_size * config.merge_size),
        c(h),
        c(config.intermediate_size)
    );
    let ci = 3 * config.patch_size * config.patch_size;
    // Both temporal slices see the same patch for a still image; add the bias once.
    let p0 = b.op(
        "stdlib.vision.linear-ref",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight"),
            b.weight("v.patch_embd.bias"),
            w("rows"),
            c(ci),
            c(h),
        ],
    );
    // Zero bias supplied as an external prepared weight.
    let p1 = b.op(
        "stdlib.vision.linear-ref",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight.1"),
            b.weight("__zero"),
            w("rows"),
            c(ci),
            c(h),
        ],
    );
    let patches = b.add(&p0, &p1);
    let pos = b.op(
        "stdlib.vision.position",
        vec![
            b.weight("v.position_embd.weight"),
            w("indices"),
            w("coefficients"),
            w("hidden-count"),
            c(h),
        ],
    );
    let mut x = b.add(&patches, &pos);
    for l in 0..config.layers {
        let prefix = format!("v.blk.{l}");
        let n = b.norm(&x, &format!("{prefix}.ln1"), "rows", h, config.eps);
        let qkv = b.linear(&n, &format!("{prefix}.attn_qkv"), "rows", h, 3 * h);
        let q = b.op(
            "stdlib.vision.slice",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(0)],
        );
        let k = b.op(
            "stdlib.vision.slice",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(h)],
        );
        let v = b.op(
            "stdlib.vision.slice",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(2 * h)],
        );
        let dim = h / config.heads;
        let q = b.op(
            "stdlib.vision.rope",
            vec![
                w(&q),
                w("positions"),
                w("frequencies"),
                w("hidden-count"),
                c(h),
                c(dim),
            ],
        );
        let k = b.op(
            "stdlib.vision.rope",
            vec![
                w(&k),
                w("positions"),
                w("frequencies"),
                w("hidden-count"),
                c(h),
                c(dim),
            ],
        );
        let a = b.op(
            "stdlib.vision.attention",
            vec![
                w(&q),
                w(&k),
                w(&v),
                w("rows"),
                w("rows"),
                c(config.heads),
                c(config.heads),
                c(dim),
                f(1. / (dim as f32).sqrt()),
                c(0),
                w("rows"),
            ],
        );
        let a = b.linear(&a, &format!("{prefix}.attn_out"), "rows", h, h);
        x = b.add(&x, &a);
        let n = b.norm(&x, &format!("{prefix}.ln2"), "rows", h, config.eps);
        let up = b.linear(
            &n,
            &format!("{prefix}.ffn_up"),
            "rows",
            h,
            config.intermediate_size,
        );
        let up = b.op("stdlib.vision.gelu-tanh", vec![w(&up), w("ff-count")]);
        let down = b.linear(
            &up,
            &format!("{prefix}.ffn_down"),
            "rows",
            config.intermediate_size,
            h,
        );
        x = b.add(&x, &down);
    }
    // LayerNorm is applied before reshaping neighbouring patches into merged rows.
    x = b.norm(&x, "v.post_ln", "rows", h, 1e-6);
    let merged = h * config.merge_size * config.merge_size;
    x = b.linear(&x, "mm.0", "merged-rows", merged, merged);
    x = b.op("stdlib.vision.gelu-erf", vec![w(&x), w("hidden-count")]);
    x = b.linear(&x, "mm.2", "merged-rows", merged, config.output_size);
    let source = b.render(&x, names, false);
    let marker = format!(
        "{}.]\n",
        (0..names.len())
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    prelude.insert_str(0, &marker);
    source.replacen(&marker, &prelude, 1)
}

fn lfm3_source(config: &VisionConfig, names: &[String]) -> String {
    let h = config.hidden_size;
    let d = h / config.heads;
    let k = d * config.kv_heads;
    let mut b = Builder {
        ops: vec![],
        weights: names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), format!("w{i}")))
            .collect(),
    };
    let mut prelude = format!(
        " ([.patches] stdlib.models.memory.ref-count-f32 [elements.])\n ({{[.elements] {}}} stdlib.models.memory.u64.div [rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.div [merged-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [hidden-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [ff-count.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [head-rows.])\n ({{[.rows] {}}} stdlib.models.memory.u64.mul [key-head-rows.])\n\n",
        c(3 * config.patch_size * config.patch_size),
        c(4),
        c(h),
        c(config.intermediate_size),
        c(config.heads),
        c(config.kv_heads)
    );
    let ci = 3 * config.patch_size * config.patch_size;
    let p0 = b.op(
        "stdlib.vision.linear-ref",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight"),
            b.weight("__zero"),
            w("rows"),
            c(ci),
            c(h),
        ],
    );
    let p1 = b.op(
        "stdlib.vision.linear-ref",
        vec![
            w("patches"),
            b.weight("v.patch_embd.weight.1"),
            b.weight("__zero"),
            w("rows"),
            c(ci),
            c(h),
        ],
    );
    let patches = b.add(&p0, &p1);
    let mut x = b.op(
        "stdlib.lfm3.vision.rms",
        vec![
            w(&patches),
            b.weight("v.embedding_norm.weight"),
            w("rows"),
            c(h),
            f(config.eps),
        ],
    );
    for i in 0..config.layers {
        let prefix = format!("v.blk.{i}");
        let n = b.op(
            "stdlib.lfm3.vision.rms",
            vec![
                w(&x),
                b.weight(&format!("{prefix}.ln1.weight")),
                w("rows"),
                c(h),
                f(config.eps),
            ],
        );
        let operator = if config.conv_layers.contains(&i) {
            let proj = b.linear(&n, &format!("{prefix}.conv2d_in_proj"), "rows", h, 3 * h);
            let spatial = b.op(
                "stdlib.lfm3.vision.spatial",
                vec![
                    w(&proj),
                    b.weight(&format!("{prefix}.conv2d_spatial.weight")),
                    w("rows"),
                    c(h),
                    w("width"),
                ],
            );
            b.linear(&spatial, &format!("{prefix}.conv2d_out_proj"), "rows", h, h)
        } else {
            let q = b.linear(&n, &format!("{prefix}.attn_q"), "rows", h, h);
            let key = b.linear(&n, &format!("{prefix}.attn_k"), "rows", h, k);
            let v = b.linear(&n, &format!("{prefix}.attn_v"), "rows", h, k);
            let q = b.op(
                "stdlib.lfm3.vision.rms",
                vec![
                    w(&q),
                    b.weight(&format!("{prefix}.attn_q_norm.weight")),
                    w("head-rows"),
                    c(d),
                    f(config.eps),
                ],
            );
            let key = b.op(
                "stdlib.lfm3.vision.rms",
                vec![
                    w(&key),
                    b.weight(&format!("{prefix}.attn_k_norm.weight")),
                    w("key-head-rows"),
                    c(d),
                    f(config.eps),
                ],
            );
            let q = b.op(
                "stdlib.vision.rope",
                vec![
                    w(&q),
                    w("positions"),
                    w("frequencies"),
                    w("hidden-count"),
                    c(h),
                    c(d),
                ],
            );
            let key = b.op(
                "stdlib.vision.rope",
                vec![
                    w(&key),
                    w("positions"),
                    w("frequencies"),
                    format!("({{{} {}}} stdlib.models.memory.u64.mul)", w("rows"), c(k)),
                    c(k),
                    c(d),
                ],
            );
            let key = b.op(
                "stdlib.lfm3.vision.repeat-kv",
                vec![
                    w(&key),
                    w("rows"),
                    c(config.heads),
                    c(config.kv_heads),
                    c(d),
                ],
            );
            let v = b.op(
                "stdlib.lfm3.vision.repeat-kv",
                vec![w(&v), w("rows"), c(config.heads), c(config.kv_heads), c(d)],
            );
            let a = b.op(
                "stdlib.vision.attention",
                vec![
                    w(&q),
                    w(&key),
                    w(&v),
                    w("rows"),
                    w("rows"),
                    c(config.heads),
                    c(config.heads),
                    c(d),
                    f(1. / (d as f32).sqrt()),
                    c(0),
                    w("rows"),
                ],
            );
            b.linear(&a, &format!("{prefix}.attn_out"), "rows", h, h)
        };
        x = b.add(&x, &operator);
        let n = b.op(
            "stdlib.lfm3.vision.rms",
            vec![
                w(&x),
                b.weight(&format!("{prefix}.ln2.weight")),
                w("rows"),
                c(h),
                f(config.eps),
            ],
        );
        let gate = b.linear(
            &n,
            &format!("{prefix}.ffn_gate"),
            "rows",
            h,
            config.intermediate_size,
        );
        let up = b.linear(
            &n,
            &format!("{prefix}.ffn_up"),
            "rows",
            h,
            config.intermediate_size,
        );
        let gated = b.op("stdlib.lfm3.vision.swiglu", vec![w(&gate), w(&up)]);
        let down = b.linear(
            &gated,
            &format!("{prefix}.ffn_down"),
            "rows",
            config.intermediate_size,
            h,
        );
        x = b.add(&x, &down);
    }
    x = b.norm(&x, "mm.input_norm", "rows", h, 1e-6);
    x = b.op(
        "stdlib.lfm3.vision.merge",
        vec![w(&x), w("rows"), c(h), w("width")],
    );
    x = b.linear(&x, "mm.0", "merged-rows", 4 * h, 4 * h);
    x = b.op("stdlib.vision.gelu-erf", vec![w(&x), w("hidden-count")]);
    x = b.linear(&x, "mm.2", "merged-rows", 4 * h, config.output_size);
    let source = b.render(&x, names, true);
    let marker = format!(
        "{}.]\n",
        (0..names.len())
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    prelude.insert_str(0, &marker);
    source.replacen(&marker, &prelude, 1)
}
