"""Tests for provider-backed model catalog discovery."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from mindforge.api import routes
from mindforge.api.schemas import LLMModelDiscoveryRequest
from mindforge.config import LLMConfig
from mindforge.services import model_discovery
from mindforge.services.model_discovery import (
    DiscoveredModel,
    ModelDiscoveryError,
    build_models_url,
    discover_models,
    parse_model_catalog,
    validate_discovery_target,
)


def test_openai_base_url_is_explicit_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_OPENAI_BASE_URL", raising=False)
    assert LLMConfig().openai_base_url == "https://api.openai.com/v1"
    assert (
        LLMConfig(openai_base_url="").get_base_url("openai")
        == "https://api.openai.com/v1"
    )


def test_build_models_url_preserves_provider_api_prefix() -> None:
    assert (
        build_models_url("https://api.openai.com/v1")
        == "https://api.openai.com/v1/models"
    )
    assert (
        build_models_url("https://api.deepseek.com")
        == "https://api.deepseek.com/models"
    )
    assert (
        build_models_url("https://models.example/v1/models")
        == "https://models.example/v1/models"
    )


def test_parse_model_catalog_deduplicates_and_bounds_results() -> None:
    models, truncated = parse_model_catalog(
        (
            b'{"data":['
            b'{"id":"model-b","owned_by":"vendor"},'
            b'{"id":"model-a"},'
            b'{"id":"model-a"}'
            b"]}"
        ),
        max_models=1,
    )

    assert models == [DiscoveredModel(id="model-a", owned_by="")]
    assert truncated is True


@pytest.mark.asyncio
async def test_discover_models_uses_bearer_auth_and_standard_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://models.example/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "chat-model", "owned_by": "vendor"},
                    {"id": "reasoning-model"},
                ]
            },
        )

    models, truncated = await discover_models(
        base_url="https://models.example/v1",
        api_key="secret",
        allow_private=False,
        timeout_seconds=5,
        max_response_bytes=4096,
        max_models=100,
        transport=httpx.MockTransport(handler),
        validate_target=False,
    )

    assert [model.id for model in models] == [
        "chat-model",
        "reasoning-model",
    ]
    assert truncated is False


@pytest.mark.asyncio
async def test_discovery_blocks_private_cloud_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def private_address(_hostname: str, _port: int) -> set[str]:
        return {"127.0.0.1"}

    monkeypatch.setattr(
        model_discovery,
        "_resolve_addresses",
        private_address,
    )

    with pytest.raises(ModelDiscoveryError, match="受限制"):
        await validate_discovery_target(
            "http://localhost:8001/v1/models",
            allow_private=False,
        )

    await validate_discovery_target(
        "http://localhost:8001/v1/models",
        allow_private=True,
    )


@pytest.mark.asyncio
async def test_discovery_blocks_link_local_targets_for_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def metadata_address(_hostname: str, _port: int) -> set[str]:
        return {"169.254.169.254"}

    monkeypatch.setattr(
        model_discovery,
        "_resolve_addresses",
        metadata_address,
    )

    with pytest.raises(ModelDiscoveryError, match="链路本地"):
        await validate_discovery_target(
            "http://metadata.invalid/models",
            allow_private=True,
        )


@pytest.mark.asyncio
async def test_model_discovery_route_uses_stored_masked_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_discover(**kwargs):
        captured.update(kwargs)
        return [DiscoveredModel(id="deepseek-chat")], False

    monkeypatch.setattr(
        routes,
        "_stored_provider_api_key",
        lambda provider: "stored-key" if provider == "deepseek" else "",
    )
    monkeypatch.setattr(routes, "discover_models", fake_discover)
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            api=SimpleNamespace(
                model_discovery_timeout_seconds=15,
                model_discovery_max_response_bytes=2048,
                model_discovery_max_models=100,
            )
        ),
    )

    response = await routes.discover_provider_models(
        LLMModelDiscoveryRequest(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            use_stored_api_key=True,
        )
    )

    assert response.count == 1
    assert response.models[0].id == "deepseek-chat"
    assert captured["api_key"] == "stored-key"
    assert captured["allow_private"] is False


@pytest.mark.asyncio
async def test_model_discovery_route_requires_cloud_api_key() -> None:
    with pytest.raises(routes.HTTPException) as exc_info:
        await routes.discover_provider_models(
            LLMModelDiscoveryRequest(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="",
                use_stored_api_key=False,
            )
        )

    assert exc_info.value.status_code == 400
