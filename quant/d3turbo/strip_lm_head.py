#!/usr/bin/env python3
"""Remove a CausalLM `lm_head.weight` from a compressed-tensors export."""
import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.dirs:
        path = directory / "model.safetensors"
        if not path.exists():
            print(f"skip {directory}: no model.safetensors")
            continue
        state = load_file(str(path))
        if state.pop("lm_head.weight", None) is None:
            print(f"{directory}: no lm_head")
            continue
        save_file(state, str(path), metadata={"format": "pt"})
        print(f"{directory}: stripped lm_head")


if __name__ == "__main__":
    main()
