from __future__ import annotations

import os


def get_openai_client():
    """Build an OpenAI-compatible client from injected environment variables."""
    from openai import OpenAI

    # Required by evaluator: use injected API_BASE_URL and API_KEY values.
    base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    return OpenAI(base_url=base_url, api_key=api_key)
