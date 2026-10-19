//! Prometheus recorder installation.

use anyhow::{Context, Result};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the process-global Prometheus recorder and return its scrape handle.
pub fn install_recorder() -> Result<PrometheusHandle> {
    PrometheusBuilder::new()
        .install_recorder()
        .context("installing Prometheus metrics recorder")
}
