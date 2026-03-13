"""
Shared Gemini vision helper for specialist nodes.
Uses the same direct REST approach as app.py for consistency.
"""
import json
import os
import re

import requests


def _extract_b64(screenshot: str) -> tuple[str, str]:
    """Return (raw_base64, mime_type) from a data URL or raw base64 string."""
    if screenshot.startswith("data:"):
        header, data = screenshot.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        return data, mime
    return screenshot, "image/jpeg"


def call_gemini_vision(prompt: str, screenshot: str, api_key: str, model: str | None = None) -> str:
    """
    Send an image + prompt to Gemini and return the raw text response.
    Raises ValueError if api_key is missing.
    Raises requests.HTTPError on API errors.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to JobAIBackEnd/.env")

    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    b64_data, mime_type = _extract_b64(screenshot)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        },
    }

    resp = requests.post(
        url,
        json=body,
        params={"key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response.
    Handles markdown code fences (```json ... ```) that models sometimes add.
    """
    text = text.strip()
    # Strip optional ```json ... ``` wrapper
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
