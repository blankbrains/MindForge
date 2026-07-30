"""Discover model IDs from OpenAI-compatible provider endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


@dataclass(frozen=True)
class DiscoveredModel:
    """One model returned by a provider's model catalog."""

    id: str
    owned_by: str = ""


class ModelDiscoveryError(RuntimeError):
    """A bounded, user-facing model discovery failure."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def build_models_url(base_url: str) -> str:
    """Append the standard model-list path to a validated API base URL."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if not path.endswith("/models"):
        path = f"{path}/models" if path else "/models"
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
        )
    )


async def _resolve_addresses(hostname: str, port: int) -> set[str]:
    def resolve() -> set[str]:
        results = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
        return {item[4][0] for item in results}

    try:
        return await asyncio.to_thread(resolve)
    except OSError as exc:
        raise ModelDiscoveryError(
            "无法解析模型服务地址，请检查 Base URL。",
            status_code=400,
        ) from exc


def _address_is_blocked(
    address: str,
    *,
    allow_private: bool,
) -> bool:
    ip = ipaddress.ip_address(address)
    if (
        ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    if allow_private:
        return False
    return not ip.is_global


async def validate_discovery_target(
    models_url: str,
    *,
    allow_private: bool,
) -> None:
    """Reject unsafe discovery destinations before opening a connection."""
    parsed = urlsplit(models_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ModelDiscoveryError(
            "Base URL 必须是有效的 HTTP(S) 地址。",
            status_code=400,
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = await _resolve_addresses(parsed.hostname, port)
    if not addresses:
        raise ModelDiscoveryError(
            "模型服务地址没有可用的网络解析结果。",
            status_code=400,
        )
    if any(
        _address_is_blocked(
            address,
            allow_private=allow_private,
        )
        for address in addresses
    ):
        raise ModelDiscoveryError(
            (
                "该 Base URL 指向受限制的网络地址。"
                if not allow_private
                else "该 Base URL 指向不允许访问的链路本地或保留地址。"
            ),
            status_code=400,
        )


def _upstream_error_message(payload: bytes, status_code: int) -> str:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = None
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            raw_message = error.get("message")
            if isinstance(raw_message, str):
                message = raw_message.strip()
        elif isinstance(error, str):
            message = error.strip()
        if not message:
            raw_message = data.get("message")
            if isinstance(raw_message, str):
                message = raw_message.strip()
    bounded = message[:300] if message else ""
    if status_code in {401, 403}:
        return bounded or "模型接口拒绝了当前 API Key。"
    if status_code == 404:
        return bounded or "模型接口没有提供标准 /models 端点。"
    if status_code == 429:
        return bounded or "模型接口请求过于频繁，请稍后重试。"
    return bounded or f"模型接口返回 HTTP {status_code}。"


def _extract_model_entry(item: Any) -> DiscoveredModel | None:
    if isinstance(item, str):
        model_id = item.strip()
        owner = ""
    elif isinstance(item, dict):
        raw_id = item.get("id") or item.get("name") or item.get("model")
        model_id = raw_id.strip() if isinstance(raw_id, str) else ""
        raw_owner = item.get("owned_by") or item.get("owner")
        owner = raw_owner.strip() if isinstance(raw_owner, str) else ""
    else:
        return None
    if (
        not model_id
        or len(model_id) > 512
        or any(ord(char) < 32 or ord(char) == 127 for char in model_id)
    ):
        return None
    return DiscoveredModel(id=model_id, owned_by=owner[:256])


def parse_model_catalog(
    payload: bytes,
    *,
    max_models: int,
) -> tuple[list[DiscoveredModel], bool]:
    """Parse standard and common compatible model-list response shapes."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelDiscoveryError(
            "模型接口返回的不是有效 JSON。",
        ) from exc

    if isinstance(data, dict):
        entries = data.get("data")
        if not isinstance(entries, list):
            entries = data.get("models")
    elif isinstance(data, list):
        entries = data
    else:
        entries = None
    if not isinstance(entries, list):
        raise ModelDiscoveryError(
            "模型接口响应中缺少 data 或 models 列表。",
        )

    models_by_id: dict[str, DiscoveredModel] = {}
    for item in entries:
        model = _extract_model_entry(item)
        if model is not None:
            models_by_id.setdefault(model.id, model)
    models = sorted(
        models_by_id.values(),
        key=lambda model: model.id.casefold(),
    )
    truncated = len(models) > max_models
    return models[:max_models], truncated


async def discover_models(
    *,
    base_url: str,
    api_key: str,
    allow_private: bool,
    timeout_seconds: float,
    max_response_bytes: int,
    max_models: int,
    transport: httpx.AsyncBaseTransport | None = None,
    validate_target: bool = True,
) -> tuple[list[DiscoveredModel], bool]:
    """Fetch and parse a provider model catalog with bounded resources."""
    models_url = build_models_url(base_url)
    if validate_target:
        await validate_discovery_target(
            models_url,
            allow_private=allow_private,
        )

    headers = {
        "Accept": "application/json",
        "User-Agent": "MindForge/1.0 model-discovery",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(
        timeout_seconds,
        connect=min(timeout_seconds, 10.0),
    )
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        ) as client:
            async with client.stream(
                "GET",
                models_url,
                headers=headers,
            ) as response:
                content_length = response.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > max_response_bytes
                ):
                    raise ModelDiscoveryError(
                        "模型列表响应超过允许大小。",
                    )
                payload = bytearray()
                async for chunk in response.aiter_bytes():
                    payload.extend(chunk)
                    if len(payload) > max_response_bytes:
                        raise ModelDiscoveryError(
                            "模型列表响应超过允许大小。",
                        )
                if 300 <= response.status_code < 400:
                    raise ModelDiscoveryError(
                        "模型接口返回了重定向，已拒绝跟随。",
                        status_code=400,
                    )
                if response.status_code >= 400:
                    status_code = (
                        429 if response.status_code == 429 else 400
                    )
                    if response.status_code >= 500:
                        status_code = 502
                    raise ModelDiscoveryError(
                        _upstream_error_message(
                            bytes(payload),
                            response.status_code,
                        ),
                        status_code=status_code,
                    )
    except ModelDiscoveryError:
        raise
    except httpx.TimeoutException as exc:
        raise ModelDiscoveryError(
            "拉取模型列表超时，请检查 Base URL 或网络连接。",
            status_code=504,
        ) from exc
    except httpx.HTTPError as exc:
        raise ModelDiscoveryError(
            "无法连接模型接口，请检查 Base URL 和网络连接。",
            status_code=502,
        ) from exc

    return parse_model_catalog(
        bytes(payload),
        max_models=max_models,
    )
