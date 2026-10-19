use anyhow::{Result, bail};
use jev_engine::{api::AppState, config::Config, telemetry};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

const HELP: &str = "\
jev-engine — JEV decision server (Rust API front for vLLM)

Usage:
  jev-engine                 serve
  jev-engine --check-config  load tokenizer/head/calibration, print a report, exit
  jev-engine --help
  jev-engine --version

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
    let state = AppState::load(config)?;
    let report = serde_json::json!({
        "status": "ok",
        "report": state.load_report,
        "choice_label_ids": state.engine.label_ids.as_slice(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn serve() -> Result<()> {
    let config = Config::from_env()?;
    let app_state = AppState::load(config)?;
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
        vllm_url = %state.config.vllm_url,
        model = %state.config.vllm_model,
        checkpoint = %state.config.checkpoint.display(),
        tokenizer = %state.load_report.tokenizer.display(),
        head = %state.load_report.head.display(),
        calibration = %state.load_report.calibration.display(),
        max_options = state.load_report.max_options,
        max_concurrency = state.config.max_concurrency,
        request_timeout_secs = state.config.request_timeout.as_secs(),
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
