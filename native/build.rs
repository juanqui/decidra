//! Compiles the CUDA kernels to sm_89 cubins.
//!
//! Two modules are produced:
//!   - `kernels.cu`: the engine's own kernels
//!   - `marlin.cu`: the vendored Marlin W4A16 kernel + repack (third_party/)
//!
//! The host may not have nvcc. Resolution order:
//!   1. `JEV_NVCC` (path to nvcc)
//!   2. `nvcc` on PATH
//!   3. `docker run` the CUDA image in `JEV_CUDA_IMAGE` (default
//!      nvidia/cuda:13.1.1-devel-ubuntu24.04) purely as a build toolchain
//!
//! Cubin paths are passed to the crate through `JEV_KERNELS_CUBIN` and
//! `JEV_MARLIN_CUBIN`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;
use std::{env, fs};

struct Job {
    name: &'static str,
    src: &'static str,
    extra: &'static [&'static str],
}

const FI_SRC: &str = "third_party/flashinfer/fi_prefill.cu";
const FI_SO: &str = "libfi_prefill.so";

const JOBS: &[Job] = &[
    Job {
        name: "kernels",
        src: "kernels/kernels.cu",
        extra: &[],
    },
    Job {
        name: "marlin",
        src: "kernels/marlin.cu",
        extra: &[
            "-I",
            "third_party",
            "-I",
            "third_party/marlin",
            "--expt-relaxed-constexpr",
        ],
    },
];

const COMMON: &[&str] = &[
    "-std=c++17",
    "-O3",
    "-fmad=false",
    "-lineinfo",
    "-cubin",
    "-arch=sm_89",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=JEV_NVCC");
    println!("cargo:rerun-if-env-changed=JEV_CUDA_IMAGE");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=third_party");

    let newest =
        newest_mtime(&manifest.join("kernels")).max(newest_mtime(&manifest.join("third_party")));

    for job in JOBS {
        let src = manifest.join(job.src);
        let cubin = out_dir.join(format!("{}.cubin", job.name));
        if let (Some(src_m), Some(out_m)) = (
            fs::metadata(&src).and_then(|m| m.modified()).ok(),
            fs::metadata(&cubin).and_then(|m| m.modified()).ok(),
        ) {
            if out_m >= src_m && newest.map(|n| out_m >= n).unwrap_or(false) {
                continue;
            }
        }
        build(job, &manifest, &out_dir, &cubin);
    }

    println!(
        "cargo:rustc-env=JEV_KERNELS_CUBIN={}",
        out_dir.join("kernels.cubin").display()
    );
    println!(
        "cargo:rustc-env=JEV_MARLIN_CUBIN={}",
        out_dir.join("marlin.cubin").display()
    );

    let fi_so = build_flashinfer(&manifest, &out_dir);
    println!("cargo:rustc-env=JEV_FI_PREFILL_SO={}", fi_so.display());
    println!("cargo:rustc-link-lib=dl");
}

fn build_flashinfer(manifest: &Path, out_dir: &Path) -> PathBuf {
    let so = out_dir.join(FI_SO);
    let src = manifest.join(FI_SRC);
    let newest = newest_mtime(&manifest.join("third_party"));
    if let (Some(src_m), Some(out_m)) = (
        fs::metadata(&src).and_then(|m| m.modified()).ok(),
        fs::metadata(&so).and_then(|m| m.modified()).ok(),
    ) {
        if out_m >= src_m && newest.map(|n| out_m >= n).unwrap_or(false) {
            return so;
        }
    }
    let args: Vec<String> = [
        "-O3",
        "-std=c++17",
        "-arch=sm_89",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-Xcompiler",
        "-fvisibility=hidden",
        "-I",
        "third_party/flashinfer/include",
        "-I",
        "third_party/flashinfer/cccl/libcudacxx/include",
        "-lcudart_static",
        "-lrt",
        "-lpthread",
        "-ldl",
        "-lstdc++",
        "-lm",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let Ok(nvcc) = env::var("JEV_NVCC") {
        run_native_cmd(Path::new(&nvcc), &args, manifest, FI_SRC, &so);
        return so;
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        run_native_cmd(Path::new("nvcc"), &args, manifest, FI_SRC, &so);
        return so;
    }
    if let Err(e) = run_docker_raw(&args, manifest, out_dir, FI_SRC, FI_SO) {
        panic!("failed to build FlashInfer prefill library: {e}");
    }
    so
}

fn run_native_cmd(nvcc: &Path, args: &[String], manifest: &Path, src: &str, out: &Path) {
    let status = Command::new(nvcc)
        .args(args)
        .arg("-o")
        .arg(out)
        .arg(src)
        .current_dir(manifest)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed for {src}");
}

fn build(job: &Job, manifest: &Path, out_dir: &Path, cubin: &Path) {
    let mut args: Vec<String> = COMMON.iter().map(|s| s.to_string()).collect();
    args.extend(job.extra.iter().map(|s| s.to_string()));

    if let Ok(nvcc) = env::var("JEV_NVCC") {
        run_native(Path::new(&nvcc), &args, manifest, job.src, cubin);
        return;
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        run_native(Path::new("nvcc"), &args, manifest, job.src, cubin);
        return;
    }
    if let Err(e) = run_docker(&args, manifest, out_dir, job, cubin) {
        panic!(
            "no nvcc found and docker toolchain failed: {e}. \
             Install CUDA or set JEV_NVCC / JEV_CUDA_IMAGE."
        );
    }
}

fn run_native(nvcc: &Path, args: &[String], manifest: &Path, src: &str, cubin: &Path) {
    let status = Command::new(nvcc)
        .args(args)
        .arg("-o")
        .arg(cubin)
        .arg(src)
        .current_dir(manifest)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed for {src}");
}

fn run_docker(
    args: &[String],
    manifest: &Path,
    out_dir: &Path,
    job: &Job,
    cubin: &Path,
) -> Result<(), String> {
    let out_name = cubin
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("kernel.cubin");
    run_docker_raw(args, manifest, out_dir, job.src, out_name)
}

fn run_docker_raw(
    args: &[String],
    manifest: &Path,
    out_dir: &Path,
    src: &str,
    out_name: &str,
) -> Result<(), String> {
    let image = env::var("JEV_CUDA_IMAGE")
        .unwrap_or_else(|_| "nvidia/cuda:13.1.1-devel-ubuntu24.04".into());
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
        .arg(format!("{}:/w:ro", manifest.display()))
        .arg("-v")
        .arg(format!("{}:/out", out_dir.display()))
        .arg("-w")
        .arg("/w")
        .arg("--entrypoint")
        .arg("/usr/local/cuda/bin/nvcc")
        .arg(&image)
        .args(args)
        .arg("-o")
        .arg(format!("/out/{out_name}"))
        .arg(format!("/w/{src}"));
    let out = cmd.output().map_err(|e| format!("docker exec: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "docker nvcc failed for {}:\nstdout: {}\nstderr: {}",
            src,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = None;
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = if meta.is_dir() {
            newest_mtime(&path)
        } else {
            meta.modified().ok()
        };
        if let Some(t) = mtime {
            newest = Some(newest.map_or(t, |n: SystemTime| n.max(t)));
        }
    }
    newest
}
