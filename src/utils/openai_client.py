import importlib
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Small wrapper around OpenAI API with env-based configuration."""

    def __init__(self) -> None:
        self.enabled = os.getenv("OPENAI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "300"))
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

        self._client = None
        if self.enabled and self.api_key:
            self._client = self._build_client()

    @property
    def available(self) -> bool:
        return self._client is not None

    def _build_client(self):
        try:
            openai_mod = importlib.import_module("openai")
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Add 'openai' to requirements.") from exc

        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return openai_mod.OpenAI(**client_kwargs)

    def analyze_claim(self, claim: str, local_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("OpenAI is not configured")

        system_prompt = (
            "You are a fact-checking assistant. Return strict JSON only with keys: "
            "summary, risk_flags, follow_up_questions, confidence_note. "
            "risk_flags and follow_up_questions must be arrays of short strings."
        )
        user_payload = {
            "claim": claim,
            "local_model_result": local_result or {},
            "task": "Provide conservative review notes, not a final truth verdict.",
        }
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=self.max_output_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned non-JSON payload; using fallback envelope.")
            parsed = {
                "summary": content,
                "risk_flags": [],
                "follow_up_questions": [],
                "confidence_note": "unparsed_response",
            }
        return parsed
