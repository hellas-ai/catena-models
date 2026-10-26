pub(crate) use crate::weights::startup_timing;
use anyhow::{Context, Result, ensure};
use catena_lang::runtime::{Artifact, MemOwn, MemRef, Runtime, Value};
pub struct Ops {
    pub runtime: Runtime,
    pub artifact: Artifact,
    pub zero: MemOwn,
}
impl Ops {
    pub fn new(extra: Option<&str>) -> Result<Self> {
        let source = super::kernels::bundle(extra.unwrap_or(""));
        let (runtime, artifact) = crate::catena::load_runtime(
            crate::catena::runtime_sources(&source),
            crate::model::configured_gpu_dialect()?,
        )
        .context("compiling Flux kernels")?;
        let zero = runtime.mem_f32(&[0.])?;
        Ok(Self {
            runtime,
            artifact,
            zero,
        })
    }
    pub fn run<'a>(&self, name: &str, args: Vec<Value<'a>>) -> Result<MemOwn> {
        let mut out = self
            .artifact
            .exec_values(name, args)
            .with_context(|| format!("executing {name}"))?;
        ensure!(out.len() == 1, "{name}: wrong output arity");
        match out.pop().unwrap() {
            Value::MemOwn(m) => Ok(m),
            v => anyhow::bail!("{name}: expected memory, got {v:?}"),
        }
    }
    pub fn copy(&self, x: MemRef<'_>, bf16: bool) -> Result<MemOwn> {
        self.run(
            if bf16 {
                "flux.copy-bf16"
            } else {
                "flux.copy-f32"
            },
            vec![x.into(), (x.byte_len() / if bf16 { 2 } else { 4 }).into()],
        )
    }
    pub fn linear<'a>(
        &self,
        x: MemRef<'a>,
        w: MemRef<'a>,
        rows: usize,
        ci: usize,
        co: usize,
    ) -> Result<MemOwn> {
        self.run(
            "flux.linear-bf16",
            vec![
                x.into(),
                w.into(),
                (rows as u64).into(),
                (ci as u64).into(),
                (co as u64).into(),
            ],
        )
    }
    pub fn slice(
        &self,
        x: MemRef<'_>,
        rows: usize,
        stride: usize,
        cols: usize,
        offset: usize,
        bf16: bool,
    ) -> Result<MemOwn> {
        self.run(
            if bf16 {
                "flux.slice-bf16"
            } else {
                "flux.slice-f32"
            },
            vec![
                x.into(),
                ((rows * cols) as u64).into(),
                (stride as u64).into(),
                (cols as u64).into(),
                (offset as u64).into(),
            ],
        )
    }
    pub fn norm(
        &self,
        x: MemRef<'_>,
        w: MemRef<'_>,
        b: MemRef<'_>,
        rows: usize,
        cols: usize,
        spatial: usize,
        kind: u64,
        affine: u64,
        eps: f32,
        bf16: bool,
    ) -> Result<MemOwn> {
        self.run(
            if bf16 {
                "flux.norm-bf16"
            } else {
                "flux.norm-f32"
            },
            vec![
                x.into(),
                w.into(),
                b.into(),
                (rows as u64).into(),
                (cols as u64).into(),
                (spatial as u64).into(),
                kind.into(),
                affine.into(),
                eps.into(),
            ],
        )
    }
    pub fn attention(
        &self,
        q: MemRef<'_>,
        k: MemRef<'_>,
        v: MemRef<'_>,
        rows: usize,
        keys: usize,
        heads: usize,
        kvheads: usize,
        dim: usize,
        causal: bool,
        valid: usize,
        bf16: bool,
    ) -> Result<MemOwn> {
        self.run(
            if bf16 {
                "flux.attention-bf16"
            } else {
                "flux.attention-f32"
            },
            vec![
                q.into(),
                k.into(),
                v.into(),
                (rows as u64).into(),
                (keys as u64).into(),
                (heads as u64).into(),
                (kvheads as u64).into(),
                (dim as u64).into(),
                (1.0 / (dim as f32).sqrt()).into(),
                (causal as u64).into(),
                (valid as u64).into(),
            ],
        )
    }
}
