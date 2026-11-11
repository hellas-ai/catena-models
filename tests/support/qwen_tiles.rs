use catena_lang::runtime::{GpuDialect, RuntimeModule};

pub fn dialect() -> GpuDialect {
    match std::env::var("CATENA_GPU_DIALECT").as_deref() {
        Ok("cuda") => GpuDialect::Cuda,
        _ => GpuDialect::Hip,
    }
}

pub fn check_native(module: &RuntimeModule, dialect: GpuDialect) {
    match dialect {
        GpuDialect::Cuda => {
            assert!(
                module
                    .source
                    .contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
            );
            assert!(!module.source.contains("catena_wmma_mma"));
        }
        GpuDialect::Hip => {
            assert!(
                module
                    .source
                    .contains("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32")
            );
            assert!(module.source.contains("catena_wmma_mma"));
        }
    }
}

pub fn check_toolkit(module: RuntimeModule, name: &str) -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!("catena-qwen-tiles-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let source = dir.join("kernel.cu");
    let ptx = dir.join("kernel.ptx");
    std::fs::write(&source, module.source)?;
    let result = std::process::Command::new("nvcc")
        .args(["-std=c++17", "-arch=sm_80", "--fmad=false", "--ptx"])
        .arg(&source)
        .arg("-o")
        .arg(&ptx)
        .output()?;
    anyhow::ensure!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    anyhow::ensure!(std::fs::read_to_string(ptx)?.contains("mma.sync.aligned.m16n8k16"));
    let cubin = dir.join("kernel.cubin");
    let result = std::process::Command::new("nvcc")
        .args([
            "-std=c++17",
            "-arch=sm_80",
            "--fmad=false",
            "--cubin",
            "-Xptxas=-v",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&cubin)
        .output()?;
    anyhow::ensure!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for line in String::from_utf8_lossy(&result.stderr).lines() {
        if line.contains("spill") || line.contains("Used ") {
            eprintln!("{name}: {line}");
        }
    }
    let result = std::process::Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(cubin)
        .output()?;
    anyhow::ensure!(result.status.success());
    anyhow::ensure!(String::from_utf8_lossy(&result.stdout).contains("HMMA"));
    let result = std::process::Command::new("nvcc")
        .args(["-std=c++17", "-arch=sm_75", "--ptx"])
        .arg(&source)
        .arg("-o")
        .arg(dir.join("unsupported.ptx"))
        .output()?;
    anyhow::ensure!(!result.status.success());
    anyhow::ensure!(
        String::from_utf8_lossy(&result.stderr)
            .contains("NVIDIA BF16 m16n8k16 MMA requires sm_80 or later")
    );
    Ok(())
}
