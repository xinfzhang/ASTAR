"""
Unified LLM Client

This module provides a single implementation of LLM API calls for all ASTAR components.
Includes retry logic, JSON extraction, and embedding support.

Usage:
    from astar.core.llm import call_llm, call_llm_json, get_embeddings

    # Text response
    response = call_llm("You are an expert.", "Analyze this report.")

    # JSON response with automatic extraction
    result = call_llm_json("Output JSON only.", "Extract data.")

    # Embeddings
    embeddings = get_embeddings(["text1", "text2"])
"""

import json
import re
import time
import requests
from typing import Any, Optional, List, Dict

from astar.core.config import get_config


def extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON from LLM output text.

    Uses multiple strategies to handle various output formats:
    1. Direct JSON parsing
    2. ```json``` code block extraction
    3. Bracket matching for nested structures
    4. Cleanup and retry

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed JSON object, or None if extraction fails.
    """
    if not text:
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract ```json``` code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Bracket matching (handles nested structures correctly)
    json_str = _find_json_object(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try array syntax
    json_str = _find_json_array(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 5: Cleanup and retry
    cleaned = re.sub(r'^[^{\[]*', '', text)
    cleaned = re.sub(r'[^}\]]*$', '', cleaned)
    if cleaned:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return None


def _find_json_object(text: str) -> Optional[str]:
    """
    Find the outermost JSON object using bracket matching.

    Properly handles nested braces and string escaping.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == '\\':
            escape = True
            continue

        if c == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _find_json_array(text: str) -> Optional[str]:
    """Find the outermost JSON array using bracket matching."""
    start = text.find('[')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == '\\':
            escape = True
            continue

        if c == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call LLM API and return raw text response.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        model: Optional model override.
        temperature: Optional temperature override.

    Returns:
        LLM response text, or empty string on failure.
    """
    cfg = get_config()

    url = f"{cfg.llm.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or cfg.llm.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature if temperature is not None else cfg.llm.temperature,
    }

    for attempt in range(1, cfg.llm.max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < cfg.llm.max_retries:
                time.sleep(cfg.llm.backoff_base * attempt)
            else:
                print(f"[LLM] Request failed after {cfg.llm.max_retries} attempts: {e}")
                return ""


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[Any]:
    """
    Call LLM API and return parsed JSON response.

    Automatically appends JSON format instruction to system prompt
    and extracts JSON from the response.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        model: Optional model override.
        temperature: Optional temperature override.

    Returns:
        Parsed JSON object, or None on failure.
    """
    # Append JSON format instruction (in Chinese for Chinese medical domain)
    system_with_json = system_prompt + "\n\nPlease ensure your output is valid JSON format."

    content = call_llm(system_with_json, user_prompt, model, temperature)
    if not content:
        return None

    return extract_json(content)


def get_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 256,
) -> List[List[float]]:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed.
        model: Optional embedding model override.
        batch_size: Maximum texts per API call.

    Returns:
        List of embedding vectors (each is a list of floats).
        Returns empty list on failure.
    """
    cfg = get_config()

    if not texts:
        return []

    url = f"{cfg.llm.base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    embed_model = model or cfg.llm.embedding_model

    all_embeddings: List[List[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": embed_model,
            "input": batch,
        }

        for attempt in range(1, 4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()["data"]
                # Sort by index to ensure correct order
                sorted_data = sorted(data, key=lambda x: x["index"])
                all_embeddings.extend([item["embedding"] for item in sorted_data])
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(5 * attempt)
                else:
                    print(f"[Embedding] Request failed: {e}")
                    # Return zeros for failed batch
                    all_embeddings.extend([[0.0] * 1536] * len(batch))

    return all_embeddings


def call_llm_with_messages(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call LLM API with a custom message list.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        model: Optional model override.
        temperature: Optional temperature override.

    Returns:
        LLM response text, or empty string on failure.
    """
    cfg = get_config()

    url = f"{cfg.llm.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or cfg.llm.model,
        "messages": messages,
        "temperature": temperature if temperature is not None else cfg.llm.temperature,
    }

    for attempt in range(1, cfg.llm.max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < cfg.llm.max_retries:
                time.sleep(cfg.llm.backoff_base * attempt)
            else:
                print(f"[LLM] Request failed after {cfg.llm.max_retries} attempts: {e}")
                return ""
