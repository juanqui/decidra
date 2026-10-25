//! d3-turbo decision server.

use anyhow::{Context, Result, bail};
use decidra::{api::AppState, backend::Backend, config::Config, telemetry};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

const HELP: &str = "\
decidra — state-once inference server for vllm-sr/d3-turbo (native CUDA)

Usage:
  decidra                 serve /v1/systemone
  decidra --check-config  validate the d3-turbo package and print its contract
  decidra --native-check  load the engine and run the state-once self-test
  decidra --help
  decidra --version

Only vllm-sr/d3-turbo prefix_bidir packages with a separate F32 readout are
supported. Configuration is read from the environment; see README.md.
";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            print!("{HELP}");
            Ok(())
        }
        Some("--version") | Some("-V") => {
            println!("decidra {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("--check-config") => check_config(),
        Some("--native-check") => native_check().await,
        Some(other) => bail!("unknown argument {other:?}; try --help"),
        None => serve().await,
    }
}

fn init_tracing() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("decidra=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn check_config() -> Result<()> {
    let config = Config::from_env()?;
    let decision_path = config.native_model.join("decision_config.json");
    let decision: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&decision_path)
            .with_context(|| format!("reading {}", decision_path.display()))?,
    )
    .with_context(|| format!("parsing {}", decision_path.display()))?;
    if decision.get("attention_mode").and_then(|v| v.as_str()) != Some("prefix_bidir") {
        bail!(
            "{} is not a prefix_bidir d3-turbo package",
            decision_path.display()
        );
    }
    let tokenizer = config
        .tokenizer
        .clone()
        .unwrap_or_else(|| config.checkpoint.join("tokenizer.json"));
    if !tokenizer.is_file() {
        bail!("tokenizer not found at {}", tokenizer.display());
    }
    if !config.native_model.join("readout.safetensors").is_file() {
        bail!(
            "readout.safetensors not found in {}",
            config.native_model.display()
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "ok",
            "model": config.model_id,
            "checkpoint": config.native_model,
            "tokenizer": tokenizer,
            "attention_mode": decision.get("attention_mode"),
            "format_id": decision.get("format_id"),
            "codes": decision.get("codes").and_then(|v| v.as_array()).map(|v| v.len()),
            "max_questions": decidra::config::MAX_QUESTIONS,
            "max_state_tokens": decidra::config::MAX_STATE_TOKENS,
            "native_marlin": config.native_marlin,
            "native_int8": config.native_int8,
            "native_int8_group_scale": config.native_int8_group_scale,
            "native_gpu_embed": config.native_gpu_embed,
        }))?
    );
    Ok(())
}

#[cfg(feature = "native")]
fn load_native(config: Config) -> Result<std::sync::Arc<dyn Backend>> {
    let backend = decidra::native_backend::NativeBackend::load(&config)
        .context("loading the native d3-turbo backend")?;
    Ok(std::sync::Arc::new(backend))
}

#[cfg(not(feature = "native"))]
fn load_native(_config: Config) -> Result<std::sync::Arc<dyn Backend>> {
    bail!(
        "this binary was built without the native backend; \
         rebuild with `cargo build --release --features native`"
    )
}

#[cfg(feature = "native")]
async fn native_check() -> Result<()> {
    let config = Config::from_env()?;
    let load_config = config.clone();
    let backend = tokio::task::spawn_blocking(move || load_native(load_config))
        .await
        .context("native backend loader task failed")??;
    let started = std::time::Instant::now();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "ok",
            "backend": backend.name(),
            "model": config.native_model,
            "state_once_self_test": true,
            "elapsed_secs": started.elapsed().as_secs_f64(),
        }))?
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
async fn native_check() -> Result<()> {
    bail!(
        "this binary was built without the native backend; \
         rebuild with `cargo build --release --features native`"
    )
}

async fn serve() -> Result<()> {
    let config = Config::from_env()?;
    let build = telemetry::BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        compiled_arch: compiled_arch(),
        native: cfg!(feature = "native"),
    };
    // Install before the engine loads so startup work and zero-init series are
    // visible from the first scrape.
    let metrics = if config.metrics_enabled {
        Some(telemetry::install(&config, &build)?)
    } else {
        None
    };
    let load_config = config.clone();
    let backend = tokio::task::spawn_blocking(move || load_native(load_config))
        .await
        .context("native backend loader task failed")??;
    let mut state = AppState::new(config.clone(), backend);
    let mut upkeep = None;
    if let Some(handle) = metrics {
        state = state.with_metrics(handle.clone());
        upkeep = Some(telemetry::spawn_upkeep(handle));
    }

    tracing::info!(
        bind = %config.bind_addr,
        model = %config.model_id,
        checkpoint = %config.native_model.display(),
        max_questions = decidra::config::MAX_QUESTIONS,
        max_state_tokens = decidra::config::MAX_STATE_TOKENS,
        "starting decidra d3-turbo"
    );
    let listener = TcpListener::bind(config.bind_addr)
        .await
        .map_err(|error| anyhow::anyhow!("binding {}: {error}", config.bind_addr))?;
    let result = axum::serve(listener, state.router())
        .with_graceful_shutdown(shutdown_signal())
        .await;
    if let Some(upkeep) = upkeep {
        upkeep.abort();
    }
    result?;
    Ok(())
}

fn compiled_arch() -> String {
    #[cfg(feature = "native")]
    {
        decidra_native::arch::GpuArch::from_compile_arch(decidra_native::COMPILED_ARCH)
            .map(|arch| arch.arch_label())
            .unwrap_or_else(|| "unknown".to_owned())
    }
    #[cfg(not(feature = "native"))]
    {
        "none".to_owned()
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "failed to install Ctrl-C handler");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::error!(%error, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
