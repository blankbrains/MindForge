"""Tests for the unified cloud and self-hosted LLM layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mindforge.config import LLMConfig
from mindforge.models.base import (
    BaseLLM,
    ChatMessage,
    ChatResult,
    LLMConfigurationError,
    LLMFactory,
    is_llm_configured,
)
from mindforge.models.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
)
from mindforge.agents.base import _estimate_cost, _estimate_cost_details


class DummyLLM(BaseLLM):
    def __init__(self, model: str) -> None:
        self.model = model

    async def chat(
        self,
        messages,
        tools=None,
        response_format=None,
        temperature=0.7,
        stream=False,
    ):
        del messages, tools, response_format, temperature, stream
        return ChatResult(content="ok", model=self.model)

    async def embed(self, texts):
        return [[1.0] for _ in texts]

    async def embed_single(self, text):
        del text
        return [1.0]


def test_factory_exposes_all_builtin_providers() -> None:
    assert set(LLMFactory.available_providers()) >= {
        "openai",
        "deepseek",
        "openai_compatible",
        "local",
    }


def test_factory_supports_custom_provider_registration() -> None:
    captured: dict[str, object] = {}

    def builder(model: str, kwargs: dict[str, object]) -> BaseLLM:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return DummyLLM(model)

    LLMFactory.register_provider("custom_test", builder)
    try:
        llm = LLMFactory.create("custom_test", "custom-model", region="cn")
        assert isinstance(llm, DummyLLM)
        assert captured == {
            "model": "custom-model",
            "kwargs": {"region": "cn"},
        }
    finally:
        LLMFactory.unregister_provider("custom_test")


def test_factory_unknown_provider_lists_available_providers() -> None:
    with pytest.raises(LLMConfigurationError) as exc_info:
        LLMFactory.create("missing-provider", "model")

    message = str(exc_info.value)
    assert "missing-provider" in message
    assert "openai_compatible" in message
    assert "local" in message


def test_unregistering_builtin_provider_restores_it() -> None:
    LLMFactory.unregister_provider("local")
    assert "local" in LLMFactory.available_providers()


def test_local_provider_initializes_without_api_key() -> None:
    llm = LLMFactory.create(
        "local",
        "qwen3",
        api_key="",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
    )

    assert isinstance(llm, OpenAICompatibleAdapter)
    assert llm.model == "qwen3"
    assert llm.client.max_retries == 0


def test_compatible_provider_rejects_missing_required_key() -> None:
    with pytest.raises(LLMConfigurationError, match="API key"):
        LLMFactory.create(
            "openai_compatible",
            "cloud-model",
            api_key="",
            base_url="https://models.example/v1",
            require_api_key=True,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "models.example/v1",
        "ftp://models.example/v1",
        "https://user:password@models.example/v1",
        "https://models.example/v1?token=secret",
    ],
)
def test_compatible_adapter_rejects_unsafe_base_urls(
    base_url: str,
) -> None:
    with pytest.raises(LLMConfigurationError, match="base URL"):
        OpenAICompatibleAdapter(
            model="model",
            api_key="key",
            base_url=base_url,
        )


def test_local_readiness_does_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        llm=LLMConfig(
            llm_provider="local",
            local_base_url="http://127.0.0.1:8001/v1",
            local_model="qwen3",
            local_api_key="",
            local_api_key_required=False,
        )
    )
    monkeypatch.setattr("mindforge.config.get_settings", lambda: settings)

    assert is_llm_configured("local") is True


def test_role_model_mapping_supports_custom_and_local_defaults() -> None:
    config = LLMConfig(
        compatible_model="cloud-default",
        compatible_planner_model="cloud-planner",
        local_model="local-default",
        local_critic_model="local-critic",
    )

    assert (
        config.get_model("planner", "openai_compatible")
        == "cloud-planner"
    )
    assert (
        config.get_model("researcher", "openai_compatible")
        == "cloud-default"
    )
    assert config.get_model("critic", "local") == "local-critic"
    assert config.get_model("synthesizer", "local") == "local-default"


def test_cost_estimation_does_not_invent_local_or_unknown_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    settings = SimpleNamespace(
        llm=LLMConfig(
            model_pricing={
                "openai:gpt-4o": {
                    "input": 2.5,
                    "output": 10.0,
                }
            }
        )
    )
    monkeypatch.setattr(
        "mindforge.agents.base.get_settings",
        lambda: settings,
    )

    assert _estimate_cost("qwen3", usage, "local") is None
    assert _estimate_cost(
        "unpriced-cloud-model",
        usage,
        "openai_compatible",
    ) is None
    assert _estimate_cost("gpt-4o", usage, "openai") > 0.0
    assert _estimate_cost_details(
        "unpriced-cloud-model",
        usage,
        "openai_compatible",
    ).status == "pricing_unconfigured"


def test_cost_estimation_applies_cached_input_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        llm=LLMConfig(
            model_pricing={
                "deepseek:model": {
                    "input": 1.0,
                    "cached_input": 0.1,
                    "output": 2.0,
                }
            }
        )
    )
    monkeypatch.setattr(
        "mindforge.agents.base.get_settings",
        lambda: settings,
    )

    estimate = _estimate_cost_details(
        "model",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_cache_hit_tokens": 400,
        },
        "deepseek",
    )

    assert estimate.status == "estimated"
    assert estimate.amount_usd == pytest.approx(0.00164)


@pytest.mark.asyncio
async def test_capability_flags_omit_unsupported_request_fields() -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="answer",
                            tool_calls=None,
                        )
                    )
                ],
                usage=None,
            )

    adapter = OpenAICompatibleAdapter(
        model="model",
        api_key="",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
        supports_tools=False,
        supports_json_mode=False,
    )
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result = await adapter.chat(
        [ChatMessage(role="user", content="test")],
        tools=[{"type": "function"}],
        response_format={"type": "json_object"},
    )

    assert isinstance(result, ChatResult)
    assert "tools" not in captured
    assert "response_format" not in captured


@pytest.mark.asyncio
async def test_streaming_tool_calls_are_aggregated() -> None:
    async def chunks():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="working",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(
                                    name="search",
                                    arguments='{"query":',
                                ),
                            )
                        ],
                    )
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='"mindforge"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return chunks()

    adapter = OpenAICompatibleAdapter(
        model="model",
        api_key="",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
    )
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    stream = await adapter.chat(
        [ChatMessage(role="user", content="test")],
        stream=True,
    )
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        "chunk",
        "tool_call",
        "done",
    ]
    assert events[1].tool_calls == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query":"mindforge"}',
            },
        }
    ]


@pytest.mark.asyncio
async def test_streaming_usage_is_returned_when_supported() -> None:
    async def chunks():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="answer",
                        tool_calls=None,
                    )
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                }
            ),
        )

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            return chunks()

    adapter = OpenAICompatibleAdapter(
        model="model",
        api_key="",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
        supports_stream_usage=True,
    )
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    stream = await adapter.chat(
        [ChatMessage(role="user", content="test")],
        stream=True,
    )
    events = [event async for event in stream]

    assert events[-1].type == "done"
    assert events[-1].usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
