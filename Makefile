CARGO ?= cargo
SHELL := /bin/bash

.PHONY: build test test-native fmt clippy native-check smoke run check-config eval

build:
	$(CARGO) build --release --locked -p decidra-native
	$(CARGO) build --release --locked --features native -p decidra

test:
	$(CARGO) test --locked

test-native:
	./scripts/native-smoke.sh

fmt:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

native-check:
	$(CARGO) run --release --locked --features native -- --native-check

smoke:
	./scripts/smoke.sh

run:
	./scripts/run-native.sh

check-config:
	$(CARGO) run --release --locked -- --check-config

eval:
	./scripts/run-eval.sh
