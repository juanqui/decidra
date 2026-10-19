CARGO ?= cargo
SHELL := /bin/bash

.PHONY: build test fmt clippy smoke run check-config eval

build:
	$(CARGO) build --release --locked

test:
	$(CARGO) test --locked

fmt:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

smoke:
	./scripts/smoke.sh

run:
	./scripts/run-server.sh

check-config:
	$(CARGO) run --release --locked -- --check-config

eval:
	./scripts/run-eval.sh
