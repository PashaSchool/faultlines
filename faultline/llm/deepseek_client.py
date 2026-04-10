"""
DeepSeek API client for feature and flow detection.

Uses OpenAI-compatible API. Structured output via JSON mode + Pydantic validation
with retry on parse failure.

DeepSeek API: https://api.deepseek.com
Models: deepseek-chat (V3), deepseek-reasoner (R1)
"""

import json
import logging
import os
import time

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.5


def _get_client(api_key: str | None = None, base_url: str | None = None):
    """Creates an OpenAI-compatible client for DeepSeek API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required for DeepSeek provider.\n"
            "Install with: pip install openai"
        )

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY or use --deepseek-key")

    return OpenAI(
        api_key=key,
        base_url=base_url or _DEFAULT_BASE_URL,
    )


def validate_deepseek(
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    base_url: str | None = None,
) -> tuple[bool, str]:
    """Validates DeepSeek API connectivity. Returns (is_valid, error_message)."""
    try:
        client = _get_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        if response.choices and response.choices[0].message.content:
            return True, ""
        return False, "Empty response from DeepSeek API"
    except ImportError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"DeepSeek API error: {e}"


def call_deepseek_parsed(
    system: str,
    prompt: str,
    response_model: type[BaseModel],
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    base_url: str | None = None,
    max_tokens: int = 16_384,
    temperature: float = 0,
) -> BaseModel | None:
    """Calls DeepSeek API with JSON mode and validates response against Pydantic model.

    Retries on JSON parse errors and transient API failures.
    Returns None on permanent failure.
    """
    try:
        client = _get_client(api_key, base_url)
    except (ImportError, ValueError) as e:
        logger.warning("DeepSeek client init failed: %s", e)
        return None

    # Append JSON schema instruction to system prompt
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    system_with_schema = (
        f"{system}\n\n"
        f"## Response format\n"
        f"You MUST respond with valid JSON matching this schema:\n"
        f"```json\n{schema}\n```\n"
        f"Return ONLY the JSON object, no markdown fences, no extra text."
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_with_schema},
                    {"role": "user", "content": prompt},
                ],
            )

            text = response.choices[0].message.content
            if not text:
                logger.warning("DeepSeek returned empty response (attempt %d/%d)", attempt + 1, _MAX_RETRIES)
                continue

            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            return response_model.model_validate_json(text)

        except ValidationError as e:
            logger.warning(
                "DeepSeek JSON validation failed (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES, str(e)[:200],
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY)
        except Exception as e:
            err_str = str(e)
            # Rate limit or transient error — retry
            if "rate" in err_str.lower() or "429" in err_str or "500" in err_str or "503" in err_str:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("DeepSeek transient error (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, _MAX_RETRIES, err_str[:100], delay)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
            else:
                logger.warning("DeepSeek permanent error: %s", err_str[:200])
                return None

    return None
