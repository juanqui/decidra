//! Compiles the CUDA kernels to cubins for the requested architecture.
//!
//! Two modules are produced:
//!   - `kernels.cu`: the engine's own kernels
//!   - `marlin.cu`: the vendored Marlin W4A16 kernel + repack (third_party/)
//!
//! Architecture defaults to sm_89 (RTX 4090 / L40S); set `DECIDRA_CUDA_ARCH`
//! (for example `sm_90`) to target another SM. Artifact names include the
//! architecture so switching does not reuse a cubin built for the wrong SM.
//!
//! The host may not have nvcc. Resolution order:
//!   1. `DECIDRA_NVCC` (path to nvcc)
//!   2. `nvcc` on PATH
//!   3. `docker run` the CUDA image in `DECIDRA_CUDA_IMAGE` (default
//!      nvidia/cuda:13.1.1-devel-ubuntu24.04) purely as a build toolchain
//!
//! Cubin paths are passed to the crate through `DECIDRA_KERNELS_CUBIN` and
//! `DECIDRA_MARLIN_CUBIN`.

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
const FI_FA3_SRC: &str = "third_party/flashinfer/fi_prefill_fa3.cu";
const FI_FA3_SO: &str = "libfi_prefill_fa3.so";

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

const COMMON: &[&str] = &["-std=c++17", "-O3", "-fmad=false", "-lineinfo", "-cubin"];

fn cuda_arch() -> String {
    env::var("DECIDRA_CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_owned())
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=DECIDRA_NVCC");
    println!("cargo:rerun-if-env-changed=DECIDRA_CUDA_IMAGE");
    println!("cargo:rerun-if-env-changed=DECIDRA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=DECIDRA_FA3");
    println!("cargo:rerun-if-env-changed=DECIDRA_FA3_SO");
    println!("cargo:rerun-if-env-changed=DECIDRA_CUTLASS_INCLUDE");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=third_party");

    let newest =
        newest_mtime(&manifest.join("kernels")).max(newest_mtime(&manifest.join("third_party")));

    let arch = cuda_arch();
    for job in JOBS {
        let src = manifest.join(job.src);
        let cubin = out_dir.join(format!("{}-{}.cubin", job.name, arch));
        if let (Some(src_m), Some(out_m)) = (
            fs::metadata(&src).and_then(|m| m.modified()).ok(),
            fs::metadata(&cubin).and_then(|m| m.modified()).ok(),
        ) {
            if out_m >= src_m && newest.map(|n| out_m >= n).unwrap_or(false) {
                continue;
            }
        }
        build(job, &manifest, &out_dir, &cubin, &arch);
    }

    println!("cargo:rustc-env=DECIDRA_COMPILED_ARCH={arch}");
    println!(
        "cargo:rustc-env=DECIDRA_KERNELS_CUBIN={}",
        out_dir.join(format!("kernels-{arch}.cubin")).display()
    );
    println!(
        "cargo:rustc-env=DECIDRA_MARLIN_CUBIN={}",
        out_dir.join(format!("marlin-{arch}.cubin")).display()
    );

    let fi_so = build_flashinfer(&manifest, &out_dir, &arch);
    println!("cargo:rustc-env=DECIDRA_FI_PREFILL_SO={}", fi_so.display());
    let fa3_so = build_fa3(&manifest, &out_dir);
    println!(
        "cargo:rustc-env=DECIDRA_FI_PREFILL_FA3_SO={}",
        fa3_so.display()
    );
    println!("cargo:rustc-link-lib=dl");
}

fn build_flashinfer(manifest: &Path, out_dir: &Path, arch: &str) -> PathBuf {
    let stem = FI_SO.strip_suffix(".so").unwrap_or(FI_SO);
    let so = out_dir.join(format!("{stem}-{arch}.so"));
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
    let mut args: Vec<String> = [
        "-O3",
        "-std=c++17",
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
    args.insert(2, format!("-arch={arch}"));
    if let Ok(nvcc) = env::var("DECIDRA_NVCC") {
        run_native_cmd(Path::new(&nvcc), &args, manifest, FI_SRC, &so);
        return so;
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        run_native_cmd(Path::new("nvcc"), &args, manifest, FI_SRC, &so);
        return so;
    }
    let so_name = so.file_name().and_then(|f| f.to_str()).unwrap_or(FI_SO);
    if let Err(e) = run_docker_raw(&args, manifest, out_dir, FI_SRC, so_name) {
        panic!("failed to build FlashInfer prefill library: {e}");
    }
    so
}

/// Build or embed the SM90a FA3 prefill library.
///
/// `DECIDRA_FA3_SO` embeds a prebuilt library (the normal path: FA3 needs
/// CUTLASS headers that are not vendored). `DECIDRA_FA3=1` compiles
/// `fi_prefill_fa3.cu` with nvcc and the directories listed in
/// `DECIDRA_CUTLASS_INCLUDE` (colon-separated). Otherwise an empty placeholder
/// is embedded so the engine's FA2 path is unchanged.
fn build_fa3(manifest: &Path, out_dir: &Path) -> PathBuf {
    let so = out_dir.join(FI_FA3_SO);
    if let Ok(prebuilt) = env::var("DECIDRA_FA3_SO") {
        fs::copy(&prebuilt, &so)
            .unwrap_or_else(|e| panic!("copying DECIDRA_FA3_SO={prebuilt}: {e}"));
        println!("cargo:rerun-if-changed={prebuilt}");
        return so;
    }
    if env::var("DECIDRA_FA3").as_deref() != Ok("1") {
        fs::write(&so, b"").expect("writing empty FA3 placeholder");
        return so;
    }
    let cutlass = env::var("DECIDRA_CUTLASS_INCLUDE")
        .expect("DECIDRA_FA3=1 requires DECIDRA_CUTLASS_INCLUDE (colon-separated)");
    let mut args: Vec<String> = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-arch=sm_90a",
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
    for path in cutlass.split(':').filter(|p| !p.is_empty()) {
        args.push("-I".to_owned());
        args.push(path.to_owned());
    }
    if let Ok(nvcc) = env::var("DECIDRA_NVCC") {
        run_native_cmd(Path::new(&nvcc), &args, manifest, FI_FA3_SRC, &so);
        return so;
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        run_native_cmd(Path::new("nvcc"), &args, manifest, FI_FA3_SRC, &so);
        return so;
    }
    panic!("DECIDRA_FA3=1 requires nvcc (set DECIDRA_NVCC)");
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

fn build(job: &Job, manifest: &Path, out_dir: &Path, cubin: &Path, arch: &str) {
    let mut args: Vec<String> = COMMON.iter().map(|s| s.to_string()).collect();
    args.push(format!("-arch={arch}"));
    args.extend(job.extra.iter().map(|s| s.to_string()));

    if let Ok(nvcc) = env::var("DECIDRA_NVCC") {
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
             Install CUDA or set DECIDRA_NVCC / DECIDRA_CUDA_IMAGE."
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
    let image = env::var("DECIDRA_CUDA_IMAGE")
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
