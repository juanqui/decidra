# tinylife: decidra metrics in Prometheus and Grafana

Date: 2026-10-25. Host: `tinylife` (192.168.1.84). The monitoring stack lives
outside this repository at `/opt/monitoring-stack` (root-owned; dashboards are
generated).

## Prometheus

Added a scrape job and an availability probe for the host-process engine:

```yaml
  - job_name: decidra
    static_configs:
      - targets: ['127.0.0.1:8899']
        labels: {service: decidra}
```

and, in the `model_probes` job:

```yaml
      - targets: ['http://127.0.0.1:8899/health']
        labels: {service: decidra}
```

`prometheus.yml` is a single-file bind mount. Edit it in place, then
`docker restart prometheus`; do not use `kill -HUP` for a replaced inode.
Backup: `prometheus.yml.bak-2026-10-25-decider-metrics`.

Verify:

```bash
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
curl -sS 'http://127.0.0.1:9090/api/v1/targets' | jq \
  '.data.activeTargets[] | select(.labels.service=="decidra") | [.labels.job,.health]'
```

## Grafana

`dashboard/models_dashboard.py` gained a **Decidra d3-turbo - native state-once
inference** row (the generated JSON has 77 panels; source is not hand-edited).
It contains:

- live worker status, answered questions/s, successful requests/s, success
  p95, queue p95, pass p95, state-cache hit ratio, GPU free;
- throughput, latency, scheduler (running/waiting/channel/capacity), cache
  usage/entries/bytes, failures/OOM/cancellations, phase p95, and GPU memory;
- the fleet request total now includes successful decidra decisions, and the
  `service="decidra"` probe appears in the status tile and availability history.

The model-inventory table remains container-based; decidra is a host process,
so it has no `container_info` row. Its VRAM is included in the hardware
dashboard's device-wide "used (all processes)" line.

Rebuild and verify:

```bash
cd /opt/monitoring-stack/dashboard
sudo python3 build_dashboards.py
cd /opt/monitoring-stack
sudo python3 tools/check_queries.py --prom http://127.0.0.1:9090 dashboard/models.json
curl -sS http://127.0.0.1:3000/api/dashboards/uid/tinylife-models | jq '.dashboard.panels | length'
```

Grafana's file provider re-reads the JSON within 30 s. Backup:
`models_dashboard.py.bak-2026-10-25-decider-metrics`.

## Evidence

- Prometheus targets `decidra` and `model_probes{service="decidra"}`: `up`.
- Dashboard query check: no errors; decidra panels return values (worker 1,
  p95s, scheduler capacity 64, cache entries/bytes, GPU memory).
- `promtool check metrics` on a live scrape: exit 0.
- Live smoke: text request, image request, GPU hit, host restore, 8 concurrent
  requests; request/engine/cache counters moved as designed.
