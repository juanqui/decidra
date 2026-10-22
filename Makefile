CARGO ?= cargo
SHELL := /bin/bash

.PHONY: build test test-native fmt clippy native-check smoke run check-config image image-push

build:
	$(CARGO) build --release --locked --features native

test:
	$(CARGO) test --locked -p decidra

test-native:
	./scripts/native-smoke.sh

fmt:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

native-check:
	./scripts/native-smoke.sh

smoke:
	./scripts/native-smoke.sh

run:
	./scripts/run-native.sh

check-config:
	$(CARGO) run --release --locked --features native -- --check-config

IMAGE ?= ghcr.io/juanqui/decidra:0.2.0
CUDA_ARCH ?= sm_89

image:
	docker build --build-arg CUDA_ARCH=$(CUDA_ARCH) -t $(IMAGE) .

image-push: image
	docker push $(IMAGE)
