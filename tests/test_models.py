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
from mindforge.models.deepseek_adapter import DeepSeekAdapter
from mindforge.models.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
)
from mindforge.models.native_search import (
    GLMWebSearchAdapter,
    KimiBuiltinSearchAdapter,
    normalize_responses_web_search,
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
        "kimi",
        "glm",
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


def test_role_model_mapping_supports_independent_provider_defaults() -> None:
    config = LLMConfig(
        kimi_model="kimi-default",
        kimi_planner_model="kimi-planner",
        glm_model="glm-default",
        glm_critic_model="glm-critic",
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
    assert config.get_model("planner", "kimi") == "kimi-planner"
    assert config.get_model("researcher", "kimi") == "kimi-default"
    assert config.get_model("critic", "glm") == "glm-critic"
    assert config.get_model("synthesizer", "glm") == "glm-default"
    assert config.get_model("critic", "local") == "local-critic"
    assert config.get_model("synthesizer", "local") == "local-default"


def test_kimi_and_glm_providers_use_isolated_compatible_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        llm=LLMConfig(
            kimi_api_key="kimi-key",
            kimi_model="kimi-k2",
            glm_api_key="glm-key",
            glm_model="glm-4.5",
        )
    )
    monkeypatch.setattr("mindforge.config.get_settings", lambda: settings)

    kimi = LLMFactory.create("kimi", "kimi-k2")
    glm = LLMFactory.create("glm", "glm-4.5")

    assert isinstance(kimi, OpenAICompatibleAdapter)
    assert isinstance(glm, OpenAICompatibleAdapter)
    assert kimi.provider_name == "kimi"
    assert glm.provider_name == "glm"
    assert isinstance(
        kimi._native_search_adapter,
        KimiBuiltinSearchAdapter,
    )
    assert isinstance(glm._native_search_adapter, GLMWebSearchAdapter)
    assert (
        glm._native_search_adapter._endpoint
        == "https://open.bigmodel.cn/api/paas/v4/web_search"
    )


def test_responses_search_normalizes_structured_and_markdown_sources() -> None:
    response = SimpleNamespace(
        output_text=(
            "See [Python](https://www.python.org/) and "
            "https://docs.python.org/3/"
        ),
        model_dump=lambda **_kwargs: {
            "model": "search-model",
            "usage": {"input_tokens": 12, "output_tokens": 4},
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Python",
                                "url": (
                                    "https://www.python.org/"
                                    "#ws_call_id=call_123"
                                ),
                            }
                        ]
                    },
                }
            ],
        },
    )

    result = normalize_responses_web_search(
        response,
        provider="test",
        max_results=5,
    )

    assert result.text.startswith("See [Python]")
    assert [source["url"] for source in result.sources] == [
        "https://www.python.org/",
        "https://docs.python.org/3/",
    ]
    assert result.usage["input_tokens"] == 12
    assert result.answer_ready is False


def test_responses_search_keeps_source_specific_evidence() -> None:
    response = SimpleNamespace(
        output_text=(
            "Python evidence from "
            "[Python docs](https://docs.python.org/3/).\n\n"
            "Java evidence from "
            "[Java docs](https://docs.oracle.com/en/java/)."
        ),
        model_dump=lambda **_kwargs: {
            "model": "search-model",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Python docs",
                                "url": "https://docs.python.org/3/",
                            },
                            {
                                "title": "Java docs",
                                "url": "https://docs.oracle.com/en/java/",
                            },
                        ]
                    },
                }
            ],
        },
    )

    result = normalize_responses_web_search(
        response,
        provider="test",
        max_results=5,
    )

    assert "Python evidence" in result.sources[0]["content"]
    assert "Java evidence" not in result.sources[0]["content"]
    assert "Java evidence" in result.sources[1]["content"]
    assert "Python evidence" not in result.sources[1]["content"]


def test_responses_search_upgrades_generic_titles_from_markdown() -> None:
    url = "https://example.com/java-or-python-for-agents.html"
    response = SimpleNamespace(
        output_text=f"[Java or Python for agents]({url})",
        model_dump=lambda **_kwargs: {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Web source",
                                "url": url,
                            }
                        ]
                    },
                }
            ],
        },
    )

    result = normalize_responses_web_search(
        response,
        provider="test",
        max_results=5,
    )

    assert result.sources[0]["title"] == "Java or Python for agents"


def test_responses_search_derives_title_when_provider_has_none() -> None:
    url = "https://example.com/agent-framework-comparison.html"
    response = SimpleNamespace(
        output_text=url,
        model_dump=lambda **_kwargs: {"output": []},
    )

    result = normalize_responses_web_search(
        response,
        provider="test",
        max_results=5,
    )

    assert result.sources[0]["title"] == (
        "example.com - agent framework comparison"
    )


@pytest.mark.asyncio
async def test_kimi_builtin_search_preserves_builtin_tool_protocol() -> None:
    requests: list[dict] = []

    class Completions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="",
                                tool_calls=[
                                    SimpleNamespace(
                                        id="search-1",
                                        type="builtin_function",
                                        function=SimpleNamespace(
                                            name="$web_search",
                                            arguments=(
                                                '{"title":"Kimi docs",'
                                                '"url":"https://platform.'
                                                'moonshot.cn/docs"}'
                                            ),
                                        ),
                                    )
                                ],
                            )
                        )
                    ],
                    usage={"prompt_tokens": 10},
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "[Kimi docs]"
                                "(https://platform.moonshot.cn/docs)"
                            ),
                            tool_calls=None,
                        )
                    )
                ],
                usage={"completion_tokens": 6},
            )

    adapter = KimiBuiltinSearchAdapter(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
        model="kimi-k2",
        provider="openai_compatible",
    )

    result = await adapter.search(
        "Kimi 联网搜索",
        max_results=5,
        max_output_tokens=300,
    )

    assert requests[1]["messages"][2]["tool_calls"][0]["type"] == (
        "builtin_function"
    )
    assert requests[1]["messages"][3]["role"] == "tool"
    assert result.sources[0]["url"] == "https://platform.moonshot.cn/docs"
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 6


@pytest.mark.asyncio
async def test_glm_web_search_normalizes_structured_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "search_result": [
                    {
                        "title": "GLM docs",
                        "link": "https://docs.bigmodel.cn/",
                        "content": "Official documentation",
                        "publish_date": "2026-08-01",
                    }
                ]
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(
        "mindforge.models.native_search.httpx.AsyncClient",
        lambda **_kwargs: Client(),
    )
    adapter = GLMWebSearchAdapter(
        api_key="key",
        provider="openai_compatible",
        endpoint="https://open.bigmodel.cn/api/paas/v4/web_search",
    )

    result = await adapter.search(
        "GLM 联网搜索",
        max_results=3,
        max_output_tokens=300,
    )

    assert captured["url"] == (
        "https://open.bigmodel.cn/api/paas/v4/web_search"
    )
    assert captured["json"]["count"] == 3
    assert result.sources == [
        {
            "index": 1,
            "title": "GLM docs",
            "url": "https://docs.bigmodel.cn/",
            "content": "Official documentation",
            "source": "web",
            "backend": "openai_compatible:native",
            "published_at": "2026-08-01",
        }
    ]


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


@pytest.mark.asyncio
async def test_deepseek_stream_exposes_reasoning_as_internal_heartbeat() -> None:
    async def chunks():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content="internal reasoning",
                        tool_calls=None,
                    )
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="answer",
                        reasoning_content=None,
                        tool_calls=None,
                    )
                )
            ],
            usage=None,
        )

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return chunks()

    adapter = DeepSeekAdapter(
        model="deepseek-reasoner",
        api_key="test-key",
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
        "heartbeat",
        "chunk",
        "done",
    ]
    assert events[1].content == "answer"


@pytest.mark.asyncio
async def test_deepseek_structured_response_uses_valid_reasoning_json() -> None:
    message = SimpleNamespace(
        content=None,
        reasoning_content=(
            "Internal analysis before the payload.\n"
            '{"scores": {"overall": 8}}'
        ),
        tool_calls=None,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )

    class Completions:
        async def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            return response

    adapter = DeepSeekAdapter(
        model="deepseek-reasoner",
        api_key="test-key",
    )
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    result = await adapter.chat(
        [ChatMessage(role="user", content="test")],
        response_format={"type": "json_object"},
    )

    assert isinstance(result, ChatResult)
    assert result.content == '{"scores": {"overall": 8}}'
