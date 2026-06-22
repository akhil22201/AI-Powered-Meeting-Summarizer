"""
All communication with the local Ollama server lives here. Every other
module that needs an LLM call goes through this client rather than calling
`requests` directly.
"""

import json

import requests

import config
from utils.logger import get_logger

log = get_logger(__name__)


def get_available_models() -> list[str]:
    """Lists models currently pulled into the local Ollama instance."""
    response = requests.get(
        f"{config.OLLAMA_SERVER_URL}/api/tags", timeout=config.OLLAMA_TIMEOUT_SECONDS
    )
    if response.status_code == 200:
        models = response.json()["models"]
        return [model["model"] for model in models]

    raise RuntimeError(f"Failed to retrieve models from Ollama server: {response.text}")


def _empty_analysis() -> dict:
    return {
        "tldr": "",
        "key_decisions": [],
        "action_items": [],
        "participants": [],
        "sentiment": {"overall": "Unknown", "explanation": ""},
        "topics": [],
    }


def _build_analysis_prompt(context: str, transcript: str) -> str:
    return f"""You are an assistant that analyzes meeting transcripts and
returns ONLY a single valid JSON object — no prose, no markdown fences,
no commentary before or after.

Context: {context if context else 'No additional context provided.'}

Transcript:
{transcript}

Return a JSON object with EXACTLY this shape:

{{
  "tldr": "2-3 sentence high-level summary",
  "key_decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"task": "...", "owner": "Not specified", "deadline": "Not specified", "priority": "High|Medium|Low"}}
  ],
  "participants": ["name or role 1", "name or role 2"],
  "sentiment": {{"overall": "Positive|Neutral|Negative|Mixed", "explanation": "one short sentence"}},
  "topics": [{{"topic": "short topic name", "description": "one short sentence"}}]
}}

Rules:
- If a list would be empty, return an empty array [] — never omit the key.
- If owner or deadline isn't mentioned for a task, use the string "Not specified".
- priority must be inferred from urgency/context; default to "Medium" if unclear.
- participants should list names if spoken, otherwise inferred roles (e.g. "Speaker 1").
- Output must be valid JSON and nothing else.
"""


def analyze_transcript(llm_model_name: str, context: str, transcript: str) -> dict:
    """Single Ollama call that returns a fully structured meeting analysis.

    Bundling summary + sentiment + topics + structured action items into
    one prompt avoids paying the latency/cost of multiple separate LLM
    round-trips. Uses Ollama's `format: "json"` mode for reliable parsing.

    Returns a dict with keys: tldr, key_decisions (list[str]),
    action_items (list[dict]), participants (list[str]),
    sentiment (dict), topics (list[dict]).

    On any parsing failure, returns a dict where every field is empty
    except `tldr`, which holds the raw model output so nothing is lost.
    """
    prompt = _build_analysis_prompt(context, transcript)

    payload = {
        "model": llm_model_name,
        "prompt": prompt,
        "format": "json",
    }

    log.info("Sending transcript to Ollama model=%s for analysis", llm_model_name)
    response = requests.post(
        f"{config.OLLAMA_SERVER_URL}/api/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=config.OLLAMA_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to analyze transcript with model {llm_model_name}: {response.text}"
        )

    full_response = ""
    try:
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line.decode("utf-8"))
                full_response += json_line.get("response", "")
                if json_line.get("done", False):
                    break
    except json.JSONDecodeError:
        log.error("Invalid JSON in Ollama stream chunks.")
        result = _empty_analysis()
        result["tldr"] = "Failed to parse server response."
        return result

    try:
        parsed = json.loads(full_response)
    except json.JSONDecodeError:
        log.warning("Model did not return valid JSON. Raw output kept in tldr.")
        result = _empty_analysis()
        result["tldr"] = full_response.strip()
        return result

    # Defensive merge: guarantee every expected key exists with the right
    # type even if the model dropped one — never let a missing key crash
    # the UI layer downstream.
    result = _empty_analysis()
    result["tldr"] = parsed.get("tldr") or ""
    result["key_decisions"] = parsed.get("key_decisions") or []
    result["action_items"] = parsed.get("action_items") or []
    result["participants"] = parsed.get("participants") or []
    result["sentiment"] = parsed.get("sentiment") or result["sentiment"]
    result["topics"] = parsed.get("topics") or []

    return result
