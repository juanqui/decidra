use anyhow::{Context, Result, bail};
use jev_engine::{
    api::{self, AppState},
    backend::Backend,
    config::Config,
    telemetry,
};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

const HELP: &str = "\
jev-engine — JEV decision server (native CUDA backend)

Usage:
  jev-engine                 serve
  jev-engine --check-config  load tokenizer/head/calibration, print a report, exit
  jev-engine --native-check  load the native engine and run one tiny GPU forward
  jev-engine --help
  jev-engine --version

`serve` and `--native-check` require a binary built with `--features native`.
Configuration is read from the environment; see README.md.
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
            println!("jev-engine {}", env!("CARGO_PKG_VERSION"));
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
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("jev_engine=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn check_config() -> Result<()> {
    let config = Config::from_env()?;
    let metadata = api::load_metadata(&config)?;
    let report = serde_json::json!({
        "status": "ok",
        "report": metadata.report,
        "choice_label_ids": metadata.label_ids,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(feature = "native")]
fn load_native(
    config: Config,
) -> Result<std::sync::Arc<jev_engine::native_backend::NativeBackend>> {
    use std::sync::Arc;

    let backend = jev_engine::native_backend::NativeBackend::load(&config)
        .context("loading the native CUDA backend")?;
    Ok(Arc::new(backend))
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
    let metadata = api::load_metadata(&config)?;
    let ids = metadata.head.verbalizer_ids[metadata.head.noul[0]..metadata.head.noul[1]].to_vec();
    let prompt_text = jev_engine::prompt::text_prompt(
        "noul",
        "native check",
        "Is the engine ready?",
        &["false".to_owned(), "true".to_owned()],
    );

    let load_config = config.clone();
    let backend = tokio::task::spawn_blocking(move || load_native(load_config))
        .await
        .context("native backend loader task failed")??;
    let started = std::time::Instant::now();
    let logits = backend.logits(&prompt_text, &ids).await?;
    backend.prompt_tokens(&prompt_text);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "ok",
            "backend": "native",
            "model": config.native_model,
            "prompt": prompt_text,
            "head_ids": ids,
            "logits": logits,
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
    let load_config = config.clone();
    let backend = tokio::task::spawn_blocking(move || load_native(load_config))
        .await
        .context("native backend loader task failed")??;
    let app_state = AppState::load(config, backend)?;
    let metrics = if app_state.config.metrics_enabled {
        Some(telemetry::install_recorder()?)
    } else {
        None
    };
    let state = match metrics {
        Some(handle) => app_state.with_metrics(handle),
        None => app_state,
    };

    tracing::info!(
        bind = %state.config.bind_addr,
        backend = %state.load_report.backend,
        native_model = %state.config.native_model.display(),
        checkpoint = %state.config.checkpoint.display(),
        tokenizer = %state.load_report.tokenizer.display(),
        head = %state.load_report.head.display(),
        calibration = %state.load_report.calibration.display(),
        max_options = state.load_report.max_options,
        max_concurrency = state.config.max_concurrency,
        signing = state.config.signing_enabled(),
        metrics = state.config.metrics_enabled,
        "starting jev-engine"
    );

    let listener = TcpListener::bind(state.config.bind_addr)
        .await
        .map_err(|error| anyhow::anyhow!("binding {}: {error}", state.config.bind_addr))?;
    axum::serve(listener, state.router())
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
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
