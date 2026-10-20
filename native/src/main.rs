use jev_native::{Engine, Stage, config, cuda, tensor};

use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("self-test");
    if cmd.starts_with("--") {
        return cmd_layer(&args);
    }
    match cmd {
        "self-test" => self_test(),
        "layer" => cmd_layer(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "dequant" => cmd_dequant(&args[1..]),
        "matmul" => cmd_matmul(&args[1..]),
        "batch" => cmd_batch(&args[1..]),
        "marlin-dump" => cmd_marlin_dump(&args[1..]),
        "alloctest" => cmd_alloctest(),
        other => bail!(
            "unknown command {other:?} (self-test|layer|compare|info|dequant|matmul|alloctest)"
        ),
    }
}

fn opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn req<'a>(args: &'a [String], flag: &str) -> Result<&'a str> {
    opt(args, flag).with_context(|| format!("missing required {flag}"))
}

fn self_test() -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use half::bf16;

    let gpu = cuda::Gpu::new(0)?;
    let (free, total) = gpu.mem_info()?;
    println!(
        "device: {} | free {:.2} GiB / total {:.2} GiB",
        gpu.device_name()?,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );

    let n = 1024usize;
    let host_in: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.001)).collect();
    let dev_in = gpu.stream.clone_htod(&host_in)?;
    let mut dev_out = gpu.stream.alloc_zeros::<bf16>(n)?;

    let f = gpu.func("self_test_kernel")?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = gpu.stream.launch_builder(&f);
    launch.arg(&dev_in);
    launch.arg(&mut dev_out);
    let n_i32 = n as i32;
    launch.arg(&n_i32);
    unsafe { launch.launch(cfg) }?;
    gpu.synchronize()?;

    let host_out: Vec<bf16> = gpu.stream.clone_dtoh(&dev_out)?;
    let mut max_err = 0f32;
    for i in 0..n {
        let want = 2.0 * host_in[i].to_f32() + 1.0;
        max_err = max_err.max((host_out[i].to_f32() - want).abs());
    }
    if max_err > 1e-2 {
        bail!("self-test kernel mismatch: max abs err {max_err}");
    }
    println!("self-test kernel OK: n={n} max_abs_err={max_err:.3e}");
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let cfg = config::Config::from_dir(&model)?;
    let store = tensor::Store::open(&model)?;
    println!("model: {}", model.display());
    println!("{cfg:#?}");
    println!("tensors: {}", store.tensor_names().count());
    Ok(())
}

fn cmd_alloctest() -> Result<()> {
    use half::bf16;
    let gpu = cuda::Gpu::new(0)?;
    for kib in [64usize, 256, 512, 640, 700, 716, 720, 724, 800, 1024, 2048] {
        let n = kib * 1024 / 2;
        let mut times = Vec::new();
        for _ in 0..6 {
            gpu.synchronize()?;
            let t = std::time::Instant::now();
            let s = gpu.stream.alloc_zeros::<bf16>(n)?;
            gpu.synchronize()?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
            drop(s);
        }
        eprintln!("{kib:5} KiB: {times:?}");
    }
    Ok(())
}

fn cmd_marlin_dump(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let tensor = req(args, "--tensor")?;
    let prefix = PathBuf::from(req(args, "--out-prefix")?);
    let eng = Engine::new(&model)?;
    eng.set_marlin(true);
    let (q, s, z) = eng.dump_marlin(tensor)?;
    let mut qb = Vec::with_capacity(q.len() * 4);
    for v in &q {
        qb.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(prefix.with_extension("q.i32"), qb)?;
    write_f32(&prefix.with_extension("s.f32"), &s)?;
    let mut zb = Vec::with_capacity(z.len() * 4);
    for v in &z {
        zb.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(prefix.with_extension("z.i32"), zb)?;
    eprintln!(
        "qweight {} ints, scales {} f32, zeros {} ints",
        q.len(),
        s.len(),
        z.len()
    );
    Ok(())
}

fn read_json_path_or_inline(s: &str) -> Result<serde_json::Value> {
    let text = if s.trim_start().starts_with('[') || s.trim_start().starts_with('{') {
        s.to_string()
    } else {
        std::fs::read_to_string(s)
            .with_context(|| format!("cannot read JSON from {s} (not inline JSON)"))?
    };
    serde_json::from_str(&text).with_context(|| format!("invalid JSON: {s}"))
}

fn parse_token_batches(v: &serde_json::Value) -> Result<Vec<Vec<u32>>> {
    let arr = v
        .as_array()
        .context("expected a JSON array of token-id arrays")?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let ids = item
            .as_array()
            .with_context(|| format!("batch item {i} is not an array"))?;
        let mut row = Vec::with_capacity(ids.len());
        for x in ids {
            let n = x
                .as_u64()
                .with_context(|| format!("batch item {i} has a non-integer token id"))?;
            row.push(n as u32);
        }
        out.push(row);
    }
    Ok(out)
}

fn cmd_batch(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let tokens_v = read_json_path_or_inline(req(args, "--tokens")?)?;
    let token_sets = parse_token_batches(&tokens_v)?;
    if token_sets.is_empty() {
        bail!("empty batch");
    }
    let shared_ids = match opt(args, "--head-ids") {
        Some(s) => Some(parse_ids(s)?),
        None => None,
    };
    let per_item_ids = match opt(args, "--head-ids-json") {
        Some(path) => {
            let v = read_json_path_or_inline(path)?;
            Some(parse_token_batches(&v)?)
        }
        None => None,
    };
    let repeat: usize = opt(args, "--repeat").unwrap_or("1").parse()?;
    let rows: Vec<&[u32]> = token_sets.iter().map(Vec::as_slice).collect();

    let eng = Engine::new(&model)?;
    // Benchmark parity with the server, which enables both chunked GDN paths.
    eng.set_gdn_chunked(true);
    eng.set_gdn_chunked_tc(true);
    if args.iter().any(|a| a == "--marlin") {
        eng.set_marlin(true);
    }
    if args.iter().any(|a| a == "--marlin-int8") {
        eng.set_marlin(true);
        eng.set_marlin_int8(true);
    }
    if let Some(v) = opt(args, "--int8-clip") {
        eng.set_int8_clip(
            v.parse()
                .context("--int8-clip must be a number in (0, 1]")?,
        );
    }
    if args.iter().any(|a| a == "--int8-group-scale") {
        eng.set_int8_group_scale(true);
    }
    if let Some(v) = opt(args, "--int8-group-exclude") {
        eng.set_int8_group_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-exclude") {
        eng.set_int8_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-all-layers") {
        eng.set_int8_all_layers(v);
    }
    if args.iter().any(|a| a == "--legacy-gdn") {
        eng.set_gdn_chunked(false);
        eng.set_gdn_chunked_tc(false);
        eng.set_gdn_shared(false);
        eng.set_gdn_split(false);
    }
    if args.iter().any(|a| a == "--gdn-shared") {
        eng.set_gdn_split(false);
    }
    if args.iter().any(|a| a == "--gdn-chunked") {
        eng.set_gdn_chunked(true);
    }
    if args.iter().any(|a| a == "--gdn-chunked-tc") {
        eng.set_gdn_chunked_tc(true);
    }
    if let Some(n) = opt(args, "--gdn-chunked-min-seq") {
        eng.set_gdn_chunked_min_seq(
            n.parse()
                .context("--gdn-chunked-min-seq must be an integer")?,
        );
    }
    if args.iter().any(|a| a == "--legacy-attention") {
        eng.set_warp_attention(false);
    }
    if args.iter().any(|a| a == "--legacy-conv") {
        eng.set_conv_tiled(false);
    }
    if args.iter().any(|a| a == "--no-cache") {
        eng.set_cache_weights(false);
    }
    if args.iter().any(|a| a == "--preload") {
        let t = std::time::Instant::now();
        eng.preload_all()?;
        eprintln!("preload: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    let total_tokens: usize = token_sets.iter().map(Vec::len).sum();
    let mut times = Vec::new();
    let mut result: Vec<Vec<f32>> = Vec::new();
    for it in 0..repeat.max(1) {
        let t = std::time::Instant::now();
        match (&shared_ids, &per_item_ids) {
            (Some(ids), _) => {
                let per: Vec<&[u32]> = (0..rows.len()).map(|_| ids.as_slice()).collect();
                result = eng.decision_logits_batch(&rows, &per)?;
            }
            (None, Some(per)) => {
                if per.len() != rows.len() {
                    bail!(
                        "--head-ids-json has {} items but the batch has {}",
                        per.len(),
                        rows.len()
                    );
                }
                let per: Vec<&[u32]> = per.iter().map(Vec::as_slice).collect();
                result = eng.decision_logits_batch(&rows, &per)?;
            }
            (None, None) => {
                result = eng.forward_last_hidden_batch(&rows)?;
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        eprintln!("batch[{it}]: {ms:.2} ms");
        if it + 1 == repeat.max(1) {
            break;
        }
    }
    if repeat > 1 {
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!(
            "batch mean {mean:.2} ms | min {min:.2} | max {max:.2} | {:.2} ms/request | {:.1} tok/s",
            mean / rows.len() as f64,
            total_tokens as f64 / (mean / 1e3)
        );
    }
    if let Some(path) = opt(args, "--out") {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        eprintln!("wrote {path}");
    }
    let is_logits = shared_ids.is_some() || per_item_ids.is_some();
    for (i, vals) in result.iter().enumerate() {
        if is_logits {
            let (max_abs, _) = stats(vals);
            println!("{i}: logits {:?} (max_abs {max_abs:.4})", vals);
        } else {
            let (max_abs, rms) = stats(vals);
            println!(
                "{i}: hidden[{}] max_abs {max_abs:.4} rms {rms:.4} first3 {:?}",
                vals.len(),
                &vals[..vals.len().min(3)]
            );
        }
    }
    Ok(())
}

fn cmd_matmul(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let tensor = req(args, "--tensor")?;
    let input = read_f32(Path::new(req(args, "--input")?))?;
    let rows: usize = req(args, "--rows")?.parse()?;
    let out = PathBuf::from(req(args, "--out")?);
    let fused = args.iter().any(|a| a == "--fused-w4a16");
    let eng = Engine::new(&model)?;
    if args.iter().any(|a| a == "--marlin") {
        eng.set_marlin(true);
    }
    if args.iter().any(|a| a == "--marlin-int8") {
        eng.set_marlin(true);
        eng.set_marlin_int8(true);
    }
    if let Some(v) = opt(args, "--int8-clip") {
        eng.set_int8_clip(
            v.parse()
                .context("--int8-clip must be a number in (0, 1]")?,
        );
    }
    if args.iter().any(|a| a == "--int8-group-scale") {
        eng.set_int8_group_scale(true);
    }
    if let Some(v) = opt(args, "--int8-group-exclude") {
        eng.set_int8_group_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-exclude") {
        eng.set_int8_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-all-layers") {
        eng.set_int8_all_layers(v);
    }
    let marlin_fp8 = args.iter().any(|a| a == "--marlin-fp8");
    let marlin_int8 = args.iter().any(|a| a == "--marlin-int8");
    if marlin_fp8 || marlin_int8 {
        eng.set_marlin(true);
        eng.set_marlin_fp8(marlin_fp8);
        eng.set_marlin_int8(marlin_int8);
    }
    if args.iter().any(|a| a == "--legacy-gdn") {
        eng.set_gdn_chunked(false);
        eng.set_gdn_chunked_tc(false);
        eng.set_gdn_shared(false);
        eng.set_gdn_split(false);
    }
    if args.iter().any(|a| a == "--gdn-shared") {
        eng.set_gdn_split(false);
    }
    if let Some(c) = opt(args, "--gdn-chunks") {
        eng.set_gdn_chunks(c.parse().context("--gdn-chunks must be an integer")?);
    }
    if let Some(n) = opt(args, "--gdn-split-min-seq") {
        eng.set_gdn_split_min_seq(
            n.parse()
                .context("--gdn-split-min-seq must be an integer")?,
        );
    }
    if args.iter().any(|a| a == "--legacy-attention") {
        eng.set_warp_attention(false);
    }
    let repeat: usize = opt(args, "--repeat").unwrap_or("1").parse()?;
    let mut vals = Vec::new();
    if args.iter().any(|a| a == "--bench") {
        let (r, f) = eng.bench_matmul(tensor, &input, rows, repeat.max(3))?;
        eprintln!("reference ms: {r:?}");
        eprintln!("fused     ms: {f:?}");
    }
    for _ in 0..repeat.max(1) {
        let t = std::time::Instant::now();
        if marlin_fp8 || marlin_int8 {
            let (got, reference) = if marlin_fp8 {
                eng.matmul_module_fp8(tensor, &input, rows)?
            } else {
                eng.matmul_module_int8(tensor, &input, rows)?
            };
            let mut max = 0f32;
            let mut sum = 0f64;
            for (a, b) in got.iter().zip(reference.iter()) {
                let d = (a - b).abs();
                max = max.max(d);
                sum += d as f64;
            }
            eprintln!(
                "w4a8 vs w4a16: max_abs {max:.6} mean_abs {:.6}",
                sum / got.len() as f64
            );
            vals = got;
        } else {
            vals = eng.matmul_module(tensor, &input, rows, fused)?;
        }
        eng.gpu.synchronize()?;
        eprintln!("matmul: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    write_f32(&out, &vals)?;
    eprintln!("wrote {} ({} values)", out.display(), vals.len());
    Ok(())
}

fn cmd_dequant(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let tensor = req(args, "--tensor")?;
    let out = PathBuf::from(req(args, "--out")?);
    let eng = Engine::new(&model)?;
    let vals = eng.dequant_module(tensor)?;
    write_f32(&out, &vals)?;
    let (max_abs, rms) = stats(&vals);
    eprintln!(
        "wrote {} ({} f32 values) max_abs={:.4} rms={:.4}",
        out.display(),
        vals.len(),
        max_abs,
        rms
    );
    Ok(())
}

fn cmd_layer(args: &[String]) -> Result<()> {
    let model = PathBuf::from(req(args, "--model")?);
    let tokens_raw = req(args, "--tokens")?;
    let out = PathBuf::from(req(args, "--out")?);
    let layer: isize = opt(args, "--layer")
        .unwrap_or("-1")
        .parse()
        .context("--layer must be an integer")?;
    let stage = Stage::parse(opt(args, "--stage").unwrap_or("layer"))?;
    let tokens = parse_tokens(tokens_raw)?;
    let input = opt(args, "--input").map(PathBuf::from);
    let last_only = args.iter().any(|a| a == "--last-token");
    let head_ids = opt(args, "--head-ids").map(parse_ids).transpose()?;
    let full_logits = args.iter().any(|a| a == "--full-logits");

    let t0 = std::time::Instant::now();
    let eng = Engine::new(&model)?;
    let init_ms = t0.elapsed().as_secs_f64() * 1e3;
    // Benchmark parity with the server, which enables both chunked GDN paths.
    eng.set_gdn_chunked(true);
    eng.set_gdn_chunked_tc(true);
    eng.set_logits_opts(head_ids.clone(), full_logits);
    if args.iter().any(|a| a == "--fused-w4a16") {
        eng.set_fused_w4a16(true);
    }
    if let Some(n) = opt(args, "--fused-min-n") {
        eng.set_fused_min_n(n.parse().context("--fused-min-n must be an integer")?);
    }
    if let Some(m) = opt(args, "--fused-max-m") {
        eng.set_fused_max_m(m.parse().context("--fused-max-m must be an integer")?);
    }
    if args.iter().any(|a| a == "--fused-tile128") {
        eng.set_fused_tile128(true);
    }
    if args.iter().any(|a| a == "--marlin") {
        eng.set_marlin(true);
    }
    if args.iter().any(|a| a == "--marlin-int8") {
        eng.set_marlin(true);
        eng.set_marlin_int8(true);
    }
    if let Some(v) = opt(args, "--int8-clip") {
        eng.set_int8_clip(
            v.parse()
                .context("--int8-clip must be a number in (0, 1]")?,
        );
    }
    if args.iter().any(|a| a == "--int8-group-scale") {
        eng.set_int8_group_scale(true);
    }
    if let Some(v) = opt(args, "--int8-group-exclude") {
        eng.set_int8_group_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-exclude") {
        eng.set_int8_exclude(
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    if let Some(v) = opt(args, "--int8-all-layers") {
        eng.set_int8_all_layers(v);
    }
    if args.iter().any(|a| a == "--legacy-gdn") {
        eng.set_gdn_chunked(false);
        eng.set_gdn_chunked_tc(false);
        eng.set_gdn_shared(false);
        eng.set_gdn_split(false);
    }
    if args.iter().any(|a| a == "--gdn-shared") {
        eng.set_gdn_split(false);
    }
    if args.iter().any(|a| a == "--gdn-chunked") {
        eng.set_gdn_chunked(true);
    }
    if args.iter().any(|a| a == "--gdn-chunked-tc") {
        eng.set_gdn_chunked_tc(true);
    }
    if let Some(n) = opt(args, "--gdn-chunked-min-seq") {
        eng.set_gdn_chunked_min_seq(
            n.parse()
                .context("--gdn-chunked-min-seq must be an integer")?,
        );
    }
    if let Some(c) = opt(args, "--gdn-chunks") {
        eng.set_gdn_chunks(c.parse().context("--gdn-chunks must be an integer")?);
    }
    if args.iter().any(|a| a == "--legacy-attention") {
        eng.set_warp_attention(false);
    }
    if args.iter().any(|a| a == "--legacy-conv") {
        eng.set_conv_tiled(false);
    }
    if args.iter().any(|a| a == "--no-cache") {
        eng.set_cache_weights(false);
    }
    let mut preload_ms = 0.0;
    if args.iter().any(|a| a == "--preload") {
        let t = std::time::Instant::now();
        eng.preload_all()?;
        preload_ms = t.elapsed().as_secs_f64() * 1e3;
        let (free, total) = eng.gpu.mem_info()?;
        eprintln!(
            "preload: {:.1} ms, {} layers, device {:.2}/{:.2} GiB free",
            preload_ms,
            eng.cfg.num_hidden_layers,
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        );
    }
    if let Some(dir) = opt(args, "--dump") {
        eng.set_dump(Path::new(dir));
    }
    let (free, total) = eng.gpu.mem_info()?;
    eprintln!(
        "device={} free={:.2}GiB/{:.2}GiB seq={} tokens layer={} stage={:?}",
        eng.gpu.device_name()?,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        tokens.len(),
        layer,
        stage
    );
    let hidden = eng.cfg.hidden_size;
    let out_cols = match stage {
        Stage::Logits => head_ids
            .as_ref()
            .map(|ids| ids.len())
            .unwrap_or(eng.cfg.vocab_size),
        _ => hidden,
    };
    let out_rows = if last_only { 1 } else { tokens.len() };
    let repeat: usize = opt(args, "--repeat")
        .unwrap_or("1")
        .parse()
        .context("--repeat must be an integer")?;
    let mut times = Vec::with_capacity(repeat);
    let mut vals = Vec::new();
    for it in 0..repeat.max(1) {
        let t = std::time::Instant::now();
        vals = match &input {
            Some(path) => {
                let h = read_f32(path)?;
                eng.forward_hidden(&h, tokens.len(), layer, stage, last_only)?
            }
            None => eng.forward(&tokens, layer, stage, last_only)?,
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        eprintln!("forward[{it}]: {ms:.2} ms");
    }
    if repeat > 1 {
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(0.0, f64::max);
        eprintln!("forward mean {mean:.2} ms | min {min:.2} | max {max:.2}");
    }
    eprintln!("init {init_ms:.1} ms, preload {preload_ms:.1} ms");
    write_f32(&out, &vals)?;
    let side = serde_json::json!({
        "model": model.display().to_string(),
        "tokens": tokens,
        "layer": layer,
        "stage": format!("{stage:?}").to_lowercase(),
        "shape": [out_rows, out_cols],
        "dtype": "f32",
        "values": vals.len(),
    });
    std::fs::write(
        out.with_extension("json"),
        serde_json::to_vec_pretty(&side)?,
    )?;
    let (max_abs, rms) = stats(&vals);
    eprintln!(
        "wrote {} ({} f32 values, shape {}x{}) max_abs={:.4} rms={:.4}",
        out.display(),
        vals.len(),
        out_rows,
        out_cols,
        max_abs,
        rms
    );
    Ok(())
}

fn parse_ids(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<u32>()
                .with_context(|| format!("bad id {x:?} in --head-ids"))
        })
        .collect()
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    let text = if s.trim_start().starts_with('[') {
        s.to_string()
    } else {
        std::fs::read_to_string(s)
            .with_context(|| format!("--tokens is not inline JSON and not a readable file: {s}"))?
    };
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("--tokens must be a JSON array or a path to one ({s})"))?;
    let arr = v.as_array().context("--tokens must be a JSON array")?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let n = x.as_u64().context("token ids must be unsigned integers")?;
        out.push(n as u32);
    }
    Ok(out)
}

fn write_f32(path: &Path, vals: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!(
            "{}: size {} is not a multiple of 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn stats(vals: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut ss = 0f64;
    for &v in vals {
        max_abs = max_abs.max(v.abs());
        ss += (v as f64) * (v as f64);
    }
    (max_abs, (ss / vals.len().max(1) as f64).sqrt() as f32)
}

fn cmd_compare(args: &[String]) -> Result<()> {
    let ref_path = PathBuf::from(req(args, "--ref")?);
    let got_path = PathBuf::from(req(args, "--got")?);
    let a = read_f32(&ref_path)?;
    let b = read_f32(&got_path)?;
    if a.len() != b.len() {
        bail!("length mismatch: ref {} vs got {}", a.len(), b.len());
    }
    let n = a.len();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_diff2 = 0f64;
    let mut sum_ref2 = 0f64;
    let mut dot = 0f64;
    let mut norm_a = 0f64;
    let mut norm_b = 0f64;
    let mut max_rel = 0f32;
    let mut max_abs_idx = 0usize;
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        let d = (x - y).abs();
        if d as f32 > max_abs {
            max_abs = d as f32;
            max_abs_idx = i;
        }
        sum_abs += d;
        sum_diff2 += d * d;
        sum_ref2 += x * x;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
        let rel = d / (x.abs() + 1e-6);
        if rel as f32 > max_rel {
            max_rel = rel as f32;
        }
    }
    let rms_diff = (sum_diff2 / n as f64).sqrt();
    let rms_ref = (sum_ref2 / n as f64).sqrt();
    let cos = dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-30);
    let result = serde_json::json!({
        "n": n,
        "max_abs": max_abs,
        "max_abs_index": max_abs_idx,
        "mean_abs": sum_abs / n as f64,
        "rms_diff": rms_diff,
        "rms_ref": rms_ref,
        "rel_rms": rms_diff / rms_ref.max(1e-30),
        "max_rel": max_rel,
        "cosine": cos,
        "ref_max_abs": a.iter().fold(0f32, |m, v| m.max(v.abs())),
        "got_max_abs": b.iter().fold(0f32, |m, v| m.max(v.abs())),
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(path) = opt(args, "--json") {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(())
}
