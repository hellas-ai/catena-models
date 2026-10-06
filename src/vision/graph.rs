//! Generate one linear-use Catena composition; Rust does not execute encoder layers.
use super::VisionConfig;
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
        self.op(
            "flux.linear-f32-orr",
            vec![
                w(x),
                self.weight(&format!("{prefix}.weight")),
                self.weight(&format!("{prefix}.bias")),
                w(rows),
                c(ci),
                c(co),
            ],
        )
    }
    fn norm(&mut self, x: &str, prefix: &str, rows: &str, cols: usize, eps: f32) -> String {
        self.op(
            "flux.norm-f32-orr",
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
        self.op("flux.add-f32-oo", vec![w(x), w(y), w("hidden-count")])
    }
    fn render(self, result: &str, names: &[String]) -> String {
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
                    " ({{{} {}}} qwen3.free-f32-before [{next}.])\n",
                    w(aliases.get(name).map(String::as_str).unwrap_or(name)),
                    w(&continuation)
                );
                continuation = next;
            }
            aliases.insert(op.out.clone(), continuation);
        }
        body += &w(aliases.get(result).map(String::as_str).unwrap_or(result));
        let types = vec!["(cap.ref mem)"; 5 + names.len()].join(" ");
        let weight_args: Vec<_> = (0..names.len()).map(|i| format!("w{i}")).collect();
        format!(
            "\n(def program vision.encode : {{{types}}} -> (cap.own mem)\n= ([patches positions indices coefficients frequencies {}.]\n{body}))\n",
            weight_args.join(" ")
        )
    }
}
pub(crate) fn source(config: &VisionConfig, names: &[String]) -> String {
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
        " ([.patches] mem.cast.f32 [patch-elements _patch-buffer.])\n ([.patch-elements] :.forget [elements.])\n ({{[.elements] {}}} u64.div [rows.])\n ({{[.rows] {}}} u64.div [merged-rows.])\n ({{[.rows] {}}} u64.mul [hidden-count.])\n ({{[.rows] {}}} u64.mul [ff-count.])\n",
        c(3 * config.patch_size * config.patch_size),
        c(config.merge_size * config.merge_size),
        c(h),
        c(config.intermediate_size)
    );
    let ci = 3 * config.patch_size * config.patch_size;
    // Both temporal slices see the same patch for a still image; add the bias once.
    let p0 = b.op(
        "flux.linear-f32",
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
        "flux.linear-f32",
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
        "vision.position",
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
            "flux.slice-f32-o",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(0)],
        );
        let k = b.op(
            "flux.slice-f32-o",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(h)],
        );
        let v = b.op(
            "flux.slice-f32-o",
            vec![w(&qkv), w("hidden-count"), c(3 * h), c(h), c(2 * h)],
        );
        let dim = h / config.heads;
        let q = b.op(
            "vision.rope-orr",
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
            "vision.rope-orr",
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
            "flux.attention-f32-ooo",
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
        let up = b.op("vision.gelu-tanh-o", vec![w(&up), w("ff-count")]);
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
    x = b.op("vision.gelu-erf-o", vec![w(&x), w("hidden-count")]);
    x = b.linear(&x, "mm.2", "merged-rows", merged, config.output_size);
    let source = b.render(&x, names);
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
