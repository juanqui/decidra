//! Compiles `kernels/*.cu` to a cubin for sm_89.
//!
//! The host may not have nvcc. Resolution order:
//!   1. `JEV_NVCC` (path to nvcc)
//!   2. `nvcc` on PATH
//!   3. `docker run` the image in `JEV_CUDA_IMAGE` (default vllm/vllm-openai:v0.31.0,
//!      used strictly as a build toolchain; nothing from it is linked or imported)
//!
//! The resulting cubin is embedded by path through `JEV_KERNELS_CUBIN`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cubin = out_dir.join("kernels.cubin");

    println!("cargo:rerun-if-changed=kernels/kernels.cu");
    println!("cargo:rerun-if-changed=kernels/common.cuh");
    println!("cargo:rerun-if-env-changed=JEV_NVCC");
    println!("cargo:rerun-if-env-changed=JEV_CUDA_IMAGE");
    println!("cargo:rustc-env=JEV_KERNELS_CUBIN={}", cubin.display());

    let src = kernels_dir.join("kernels.cu");
    let nvcc_args = [
        "-std=c++17",
        "-O3",
        "-fmad=false",
        "-lineinfo",
        "-cubin",
        "-arch=sm_89",
        "-o",
    ];

    // Fresh cubin if sources are older than the output.
    if cubin.exists() {
        let out_m = fs::metadata(&cubin).and_then(|m| m.modified()).ok();
        let src_m = fs::metadata(&src).and_then(|m| m.modified()).ok();
        let cu_m = fs::metadata(kernels_dir.join("common.cuh"))
            .and_then(|m| m.modified())
            .ok();
        if let (Some(o), Some(s), Some(c)) = (out_m, src_m, cu_m) {
            if o >= s && o >= c {
                return;
            }
        }
    }

    if let Ok(nvcc) = env::var("JEV_NVCC") {
        run_native(Path::new(&nvcc), &nvcc_args, &src, &cubin);
        return;
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        run_native(Path::new("nvcc"), &nvcc_args, &src, &cubin);
        return;
    }
    if let Err(e) = run_docker(&kernels_dir, &out_dir, &nvcc_args) {
        panic!(
            "no nvcc found and docker toolchain failed: {e}. \
             Install CUDA or set JEV_NVCC / JEV_CUDA_IMAGE."
        );
    }
}

fn run_native(nvcc: &Path, args: &[&str], src: &Path, cubin: &Path) {
    let status = Command::new(nvcc)
        .args(args)
        .arg(cubin)
        .arg(src)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed");
}

fn run_docker(kernels_dir: &Path, out_dir: &Path, nvcc_args: &[&str]) -> Result<(), String> {
    let image = env::var("JEV_CUDA_IMAGE").unwrap_or_else(|_| "vllm/vllm-openai:v0.31.0".into());
    let uid = Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "0".into());
    let gid = Command::new("id")
        .arg("-g")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "0".into());

    let mut cmd = Command::new("docker");
    cmd.arg("run")
        .arg("--rm")
        .arg("--user")
        .arg(format!("{uid}:{gid}"))
        .arg("-v")
        .arg(format!("{}:/kernels:ro", kernels_dir.display()))
        .arg("-v")
        .arg(format!("{}:/out", out_dir.display()))
        .arg("-w")
        .arg("/kernels")
        .arg("--entrypoint")
        .arg("/usr/local/cuda/bin/nvcc")
        .arg(&image)
        .args(nvcc_args)
        .arg("/out/kernels.cubin")
        .arg("/kernels/kernels.cu");
    let out = cmd.output().map_err(|e| format!("docker exec: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "docker nvcc failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}
