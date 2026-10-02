"""OpenAI-compatible chat-completions provider.

Implemented with the standard library only, so it works against any endpoint
that speaks the OpenAI ``/chat/completions`` protocol: OpenAI, Azure, OpenRouter,
Ollama, LM Studio, vLLM, ...
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from ..errors import ProviderError
from ..logging_utils import get_logger
from ..messages import LLMResponse, ToolCall
from .base import LLMProvider

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ProviderConfig


def _sanitize(value: Any) -> Any:
    """Make a value safe to JSON-encode for strict API gateways.

    Message content is assembled from user prompts, model output and tool
    results (file contents, shell output, ...), any of which can carry bytes
    that a strict ``/chat/completions`` endpoint rejects with HTTP 422:

    * lone surrogates (e.g. ``\\ud800``) cannot be encoded as UTF-8 at all;
    * raw control characters other than ``\\t``, ``\\n`` and ``\\r`` are not
      valid inside JSON strings per RFC 8259.

    Both are normalised here so the request body is always valid UTF-8 JSON.
    """

    if isinstance(value, str):
        # Drop lone surrogates by round-tripping through UTF-8.
        cleaned = value.encode("utf-8", "replace").decode("utf-8")
        return "".join(
            ch if ch in "\t\n\r" or ord(ch) >= 0x20 else f"\\u{ord(ch):04x}"
            for ch in cleaned
        )
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 60.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        options = dict(options or {})
        self.extra_headers: Dict[str, str] = options.pop("extra_headers", {}) or {}
        self.options = options
        self.logger = get_logger("provider.openai")

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: "ProviderConfig") -> "OpenAICompatibleProvider":
        return cls(
            base_url=config.base_url,
            model=config.model,
            api_key=config.resolve_api_key(),
            temperature=config.temperature,
            timeout=config.timeout,
            options=config.options,
        )

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    # -- request building -------------------------------------------------- #
    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        options = dict(self.options)
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload.setdefault("tool_choice", options.pop("tool_choice", "auto"))
        payload.update(options)
        return payload

    # -- interface --------------------------------------------------------- #
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        payload = _sanitize(self._build_payload(messages, tools))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(), data=body, headers=self._headers(), method="POST"
        )
        self.logger.debug(
            "POST %s model=%s messages=%d tools=%d",
            self._endpoint(),
            self.model,
            len(messages),
            len(tools or []),
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"HTTP {exc.code} from {self._endpoint()}: {detail[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Failed to reach {self._endpoint()}: {exc.reason}"
            ) from exc

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Provider returned invalid JSON: {raw_body[:500]}"
            ) from exc
        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"Provider response has no choices: {str(data)[:500]}")
        message = choices[0].get("message") or {}

        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if content is not None and not isinstance(content, str):
            content = str(content)

        tool_calls: List[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            fn = raw_call.get("function") or {}
            args_raw = fn.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw) if args_raw.strip() else {}
                except json.JSONDecodeError as exc:
                    raise ProviderError(
                        f"Tool call '{fn.get('name')}' has invalid JSON arguments: "
                        f"{args_raw!r}"
                    ) from exc
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            if not isinstance(args, dict):
                raise ProviderError(
                    f"Tool call '{fn.get('name')}' arguments must be an object, "
                    f"got {type(args).__name__}"
                )
            tool_calls.append(
                ToolCall(
                    id=raw_call.get("id") or f"call_{uuid4().hex[:8]}",
                    name=fn.get("name") or "",
                    arguments=args,
                )
            )

        return LLMResponse(content=content, tool_calls=tool_calls, raw=data)
