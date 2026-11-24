pub(crate) use crate::weights::startup_timing;
use anyhow::{Context, Result, ensure};
use catena_lang::runtime::{Artifact, MemOwn, Runtime, Value};
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
}
