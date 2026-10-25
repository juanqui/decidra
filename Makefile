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

CUDA_ARCH ?= sm_89
VERSION ?= 0.2.0
ARCH_TAG := $(subst _,,$(CUDA_ARCH))
IMAGE ?= ghcr.io/juanqui/decidra:$(ARCH_TAG)-$(VERSION)

image:
	docker build --build-arg CUDA_ARCH=$(CUDA_ARCH) -t $(IMAGE) .

image-push: image
	docker push $(IMAGE)
