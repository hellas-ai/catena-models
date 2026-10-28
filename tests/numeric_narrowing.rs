use catena_lang::runtime::{GpuDialect, Runtime, Value};
use catena_models::catena::{compile_sources, core_sources};
const SOURCE: &str =
    "(def program narrow : (u64 val) -> (u32 val) = ([x.] [.x] stdlib.numeric.u64_to_u32))";
#[test]
fn narrowing_is_checked_and_uses_a_native_cast() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let m = compile_sources(core_sources().chain([SOURCE]), dialect)?;
        assert!(m.source.contains("<= UINT32_MAX"));
        assert!(m.source.contains("uint32_t("));
    }
    Ok(())
}
#[test]
#[ignore = "requires HIP GPU and ROCm compiler"]
fn narrowing_boundary_values() -> anyhow::Result<()> {
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let module = compile_sources(core_sources().chain([SOURCE]), GpuDialect::Hip)?;
    let artifact = runtime.load(module)?;
    for n in [0_u64, 1, 255, 256, 65536, u64::from(u32::MAX)] {
        let [Value::U32(got)] = artifact.exec("narrow", [n.into()])? else {
            panic!("wrong result type")
        };
        assert_eq!(u64::from(got), n);
    }
    Ok(())
}
