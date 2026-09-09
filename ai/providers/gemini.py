from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

from ai.providers.base import (
    ProviderConfig,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredT,
    run_provider_request,
    validate_generation_request,
)


def _reject_duplicate_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")

        result[key] = value

    return result


def _build_structured_prompt(
    prompt: str,
    system_prompt: str | None,
    response_model: type[StructuredT],
) -> str:
    """
    Build one ordinary text prompt.

    We intentionally do not use Gemini response_schema,
    response_json_schema, or response_mime_type.

    Gemini only has to generate text. The returned JSON is validated
    locally using the original Pydantic model.
    """

    schema = response_model.model_json_schema()

    schema_json = json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    parts: list[str] = []

    if system_prompt and system_prompt.strip():
        parts.append(
            "SYSTEM INSTRUCTIONS:\n"
            + system_prompt.strip()
        )

    parts.append(
        "OUTPUT REQUIREMENTS:\n"
        "Return exactly one valid JSON object matching the supplied schema.\n"
        "Return JSON only.\n"
        "Do not use Markdown.\n"
        "Do not use ``` code fences.\n"
        "Do not include explanations before or after the JSON.\n"
        "Use exact field names and enum values.\n"
        "Do not invent facts merely to fill fields.\n"
        "Respect nulls, arrays, objects, booleans and numbers according "
        "to the schema.\n\n"
        "TARGET JSON SCHEMA:\n"
        + schema_json
    )

    parts.append(
        "TASK:\n"
        + prompt.strip()
    )

    return "\n\n".join(parts)


def _parse_json_response(text: str) -> Any:
    """
    Parse one JSON value from a model response.

    Handles accidental Markdown fences conservatively while still
    rejecting trailing non-JSON content and duplicate object keys.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned[5:].lstrip()

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
    )

    # Prefer parsing the entire response directly.
    try:
        value, end = decoder.raw_decode(cleaned)

        if cleaned[end:].strip():
            raise ValueError(
                "Unexpected text after JSON payload"
            )

        return value

    except json.JSONDecodeError:
        pass

    # Conservative recovery if Gemini accidentally places a short prefix
    # before the JSON despite instructions.
    starts = [
        index
        for index in (
            cleaned.find("{"),
            cleaned.find("["),
        )
        if index >= 0
    ]

    if not starts:
        raise ValueError(
            "Gemini response does not contain JSON"
        )

    start = min(starts)

    value, end = decoder.raw_decode(
        cleaned[start:]
    )

    trailing = cleaned[start + end :].strip()

    if trailing and trailing != "```":
        raise ValueError(
            "Unexpected text after JSON payload"
        )

    return value


class GeminiProvider:
    def __init__(
        self,
        config: ProviderConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ProviderConfig.from_env()
        )

        if not self.config.api_key:
            raise ProviderConfigurationError(
                "An API key is required for the Gemini provider",
                provider="gemini",
            )

        if not self.config.model:
            raise ProviderConfigurationError(
                "A Gemini model must be configured",
                provider="gemini",
            )

        model = self.config.model.strip()

        # The current Google SDK examples use bare model IDs such as
        # gemini-2.5-flash-lite rather than models/gemini-2.5-flash-lite.
        if model.startswith("models/"):
            model = model[len("models/") :]

        if not model:
            raise ProviderConfigurationError(
                "A valid Gemini model must be configured",
                provider="gemini",
            )

        try:
            from google import genai
            from google.genai import errors
        except ImportError as exc:
            raise ProviderConfigurationError(
                "The google-genai package could not be imported correctly",
                provider="gemini",
            ) from exc

        self._genai = genai
        self._errors = errors
        self._model = model

        self._client = genai.Client(
            api_key=self.config.api_key,
        )

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_timeout_seconds(self) -> float:
        return self.config.retry.timeout_seconds

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredT],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> StructuredT:
        validate_generation_request(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            response_model=response_model,
        )

        structured_prompt = _build_structured_prompt(
            prompt,
            system_prompt,
            response_model,
        )

        # IMPORTANT:
        #
        # No response_schema.
        # No response_json_schema.
        # No response_mime_type.
        # No system_instruction.
        #
        # This is deliberately a plain text Gemini request.
        generation_config = (
            self._genai.types.GenerateContentConfig(
                temperature=temperature,
            )
        )

        response = await self._request(
            structured_prompt,
            generation_config,
            timeout_seconds,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ProviderResponseError(
                "Gemini returned an empty structured response",
                provider=self.name,
            )

        try:
            payload = _parse_json_response(text)

            return response_model.model_validate(
                payload
            )

        except Exception as exc:
            raise ProviderResponseError(
                "Gemini returned a response that did not satisfy the required schema",
                provider=self.name,
            ) from exc

    async def generate_structured_with_images(
        self,
        prompt: str,
        response_model: type[StructuredT],
        *,
        images: Sequence[tuple[bytes, str]],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> StructuredT:
        validate_generation_request(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            response_model=response_model,
        )

        if not 2 <= len(images) <= 5:
            raise ValueError(
                "Between 2 and 5 images are required"
            )

        allowed_mime_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }

        structured_prompt = _build_structured_prompt(
            prompt,
            system_prompt,
            response_model,
        )

        parts: list[Any] = [
            self._genai.types.Part.from_text(
                text=structured_prompt
            )
        ]

        for index, image in enumerate(
            images,
            start=1,
        ):
            if (
                not isinstance(image, tuple)
                or len(image) != 2
            ):
                raise TypeError(
                    f"Image {index} must be a "
                    "(bytes, mime_type) tuple"
                )

            data, mime_type = image

            if (
                not isinstance(
                    data,
                    (bytes, bytearray),
                )
                or not data
            ):
                raise ValueError(
                    f"Image {index} must contain image bytes"
                )

            if not isinstance(mime_type, str):
                raise TypeError(
                    f"Image {index} MIME type must be a string"
                )

            normalized_mime = mime_type.strip().lower()

            if normalized_mime == "image/jpg":
                normalized_mime = "image/jpeg"

            if normalized_mime not in allowed_mime_types:
                raise ValueError(
                    f"Unsupported MIME type for image {index}: "
                    f"{normalized_mime}"
                )

            parts.append(
                self._genai.types.Part.from_bytes(
                    data=bytes(data),
                    mime_type=normalized_mime,
                )
            )

        generation_config = (
            self._genai.types.GenerateContentConfig(
                temperature=temperature,
            )
        )

        response = await self._request(
            parts,
            generation_config,
            timeout_seconds,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ProviderResponseError(
                "Gemini returned an empty multimodal "
                "structured response",
                provider=self.name,
            )

        try:
            payload = _parse_json_response(text)

            return response_model.model_validate(
                payload
            )

        except Exception as exc:
            raise ProviderResponseError(
                "Gemini returned a multimodal response "
                "that did not satisfy the required schema",
                provider=self.name,
            ) from exc

    async def _request(
        self,
        contents: Any,
        generation_config: Any,
        timeout_seconds: float | None,
    ) -> Any:
        async def operation(
            _remaining: float,
        ) -> Any:
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=generation_config,
            )

        return await run_provider_request(
            operation,
            provider_name=self.name,
            retry=self.config.retry,
            timeout_seconds=timeout_seconds,
            normalize_error=self._normalize_error,
        )

    def _normalize_error(
        self,
        exc: Exception,
    ) -> ProviderError:
        api_error = getattr(
            self._errors,
            "APIError",
            None,
        )

        if (
            api_error is not None
            and isinstance(exc, api_error)
        ):
            code = getattr(
                exc,
                "code",
                None,
            )

            if code is None:
                code = getattr(
                    exc,
                    "status_code",
                    None,
                )

            if code == 429:
                return ProviderRateLimitError(
                    "Gemini rate limit was exceeded",
                    provider=self.name,
                )

            if code in {408, 504}:
                return ProviderTimeoutError(
                    "Gemini request timed out",
                    provider=self.name,
                )

            if (
                code == 409
                or (
                    isinstance(code, int)
                    and code >= 500
                )
            ):
                return ProviderUnavailableError(
                    "Gemini is temporarily unavailable",
                    provider=self.name,
                )

            if code in {401, 403}:
                return ProviderConfigurationError(
                    "Gemini API authentication or permission configuration is invalid",
                    provider=self.name,
                )

            if code == 404:
                return ProviderConfigurationError(
                    (
                        f"Gemini model {self._model!r} was not found "
                        "or is not available to this API key"
                    ),
                    provider=self.name,
                )

            if code == 400:
                return ProviderConfigurationError(
                    (
                        f"Gemini rejected the request for model "
                        f"{self._model!r}. Verify JANSETU_AI_MODEL "
                        "is a Gemini generateContent model."
                    ),
                    provider=self.name,
                )

            return ProviderError(
                "Gemini rejected the request",
                provider=self.name,
            )

        if isinstance(
            exc,
            (
                ConnectionError,
                OSError,
            ),
        ):
            return ProviderUnavailableError(
                "Gemini could not be reached",
                provider=self.name,
            )

        return ProviderError(
            "Gemini request failed",
            provider=self.name,
        )