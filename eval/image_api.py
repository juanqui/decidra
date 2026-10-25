#!/usr/bin/env python3
"""Smoke-test the d3 image API against a running Decidra server.

    python3 eval/image_api.py --url http://127.0.0.1:8899

Covers the formats/resolutions the server accepts, one malformed image, two
images in one request, and the image-cache identity rule: identical state text
with different images must not reuse the previous image's state.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

QUESTION = {
    "type": "choice",
    "instructions": "What color is the square?",
    "criteria": {"red": "red", "green": "green", "blue": "blue"},
}


def data_url(image: Image.Image, fmt: str) -> str:
    out = io.BytesIO()
    image.save(out, format=fmt)
    subtype = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{subtype};base64," + base64.b64encode(out.getvalue()).decode()


def square(color: str, width: int = 80, height: int = 60, mode: str = "RGB"):
    if mode == "P":
        rgb = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(rgb)
        box = (10, 10, width - 10, height - 10) if min(width, height) > 20 else (0, 0, width, height)
        draw.rectangle(box, fill=color)
        return rgb.convert("P")
    base = "white" if mode == "RGB" else 128 if mode == "L" else (255, 255, 255, 255)
    image = Image.new(mode, (width, height), base)
    draw = ImageDraw.Draw(image)
    box = (10, 10, width - 10, height - 10) if min(width, height) > 20 else (0, 0, width, height)
    draw.rectangle(box, fill=color)
    return image


def post(url: str, state, images: list[str]):
    body = json.dumps(
        {"state": state, "questions": {"q": QUESTION}, "images": images}
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8899")
    args = parser.parse_args()

    state = "An image from a catalogue."
    checks = 0
    for fmt in ("PNG", "JPEG", "WEBP"):
        for color in ("red", "green", "blue"):
            status, body = post(args.url, state, [data_url(square(color), fmt)])
            assert status == 200, (fmt, color, status, body)
            assert body["answers"]["q"]["choice"] == color, (fmt, color, body)
            checks += 1
    for size in ((1, 1), (31, 31), (33, 47), (1000, 100), (2000, 10)):
        status, body = post(args.url, state, [data_url(square("red", *size), "PNG")])
        assert status == 200, (size, status, body)
        checks += 1
    for mode in ("RGBA", "L", "P"):
        status, body = post(args.url, state, [data_url(square("red", mode=mode), "PNG")])
        assert status == 200, (mode, status, body)
        checks += 1
    status, body = post(
        args.url, state, [data_url(square("red"), "PNG"), data_url(square("blue"), "PNG")]
    )
    assert status == 200 and body["answers"]["q"]["choice"] in ("red", "blue"), body
    checks += 1
    status, body = post(
        args.url,
        [
            {"type": "text", "text": state},
            {"type": "image_url", "image_url": {"url": data_url(square("green"), "PNG")}},
        ],
        [],
    )
    assert status == 200 and body["answers"]["q"]["choice"] == "green", body
    checks += 1
    status, _ = post(args.url, state, ["data:image/png;base64,!!!!"])
    assert status == 400, status
    checks += 1
    print(f"image API smoke tests passed: {checks} requests")


if __name__ == "__main__":
    main()
