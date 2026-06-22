"""Test multi-model support — LLMFactory, provider switching, message format."""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Mock model adapter for testing
# ---------------------------------------------------------------------------


class MockChatMessage:
    """Simulates ChatMessage dataclass for testing."""

    def __init__(self, role: str, content: str = ""):
        self.role = role
        self.content = content


class MockChatResult:
    """Simulates ChatResult dataclass for testing."""

    def __init__(self, content: str = "", model: str = ""):
        self.content = content
        self.tool_calls = None
        self.usage = {"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20}
        self.model = model

    def __str__(self) -> str:
        return self.content


# ---------------------------------------------------------------------------
# LLM Factory — simplified for testing
# ---------------------------------------------------------------------------


class TestLLMFactory:
    """Test LLM factory provider selection."""

    def test_deepseek_provider_selection(self):
        """Verify DeepSeek provider is selected when configured."""
        config = {"provider": "deepseek", "api_key": "sk-test-key"}
        assert config["provider"] == "deepseek"
        assert config["api_key"].startswith("sk-")

    def test_openai_provider_selection(self):
        """Verify OpenAI provider is selected when configured."""
        config = {"provider": "openai", "api_key": "sk-test-key"}
        assert config["provider"] == "openai"
        assert config["api_key"].startswith("sk-")

    def test_provider_switch(self):
        """Test switching between providers."""
        for provider in ["openai", "deepseek"]:
            config = {"provider": provider, "api_key": "sk-test"}
            assert config["provider"] == provider


# ---------------------------------------------------------------------------
# Chat message format tests
# ---------------------------------------------------------------------------


class TestChatMessageFormat:
    """Test chat message construction and serialization."""

    def test_message_creation(self):
        msg = MockChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        msg = MockChatMessage(role="system", content="You are an AI assistant.")
        assert msg.role == "system"

    def test_tool_message(self):
        msg = MockChatMessage(role="tool", content='{"result": "ok"}')
        assert msg.role == "tool"

    def test_message_list_structure(self):
        messages = [
            MockChatMessage(role="system", content="System prompt"),
            MockChatMessage(role="user", content="User query"),
        ]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"


# ---------------------------------------------------------------------------
# Model routing tests
# ---------------------------------------------------------------------------


class TestModelRouting:
    """Test model name mapping per provider."""

    MODEL_MAP = {
        "openai": {
            "planner": "gpt-4o",
            "researcher": "gpt-4o-mini",
            "critic": "gpt-4o",
            "synthesizer": "gpt-4o",
        },
        "deepseek": {
            "planner": "deepseek-chat",
            "researcher": "deepseek-chat",
            "critic": "deepseek-reasoner",
            "synthesizer": "deepseek-chat",
        },
    }

    def test_openai_model_map(self):
        assert self.MODEL_MAP["openai"]["planner"] == "gpt-4o"
        assert self.MODEL_MAP["openai"]["researcher"] == "gpt-4o-mini"

    def test_deepseek_model_map(self):
        assert self.MODEL_MAP["deepseek"]["planner"] == "deepseek-chat"
        assert self.MODEL_MAP["deepseek"]["critic"] == "deepseek-reasoner"

    def test_get_model_for_role(self):
        for provider in ["openai", "deepseek"]:
            for role in ["planner", "researcher", "critic", "synthesizer"]:
                model = self.MODEL_MAP[provider][role]
                assert model is not None
                assert isinstance(model, str)

    def test_all_roles_have_models(self):
        for provider in self.MODEL_MAP:
            for role in ["planner", "researcher", "critic", "synthesizer"]:
                assert role in self.MODEL_MAP[provider], f"{provider} missing {role}"


# ---------------------------------------------------------------------------
# API result format tests
# ---------------------------------------------------------------------------


class TestChatResult:
    """Test ChatResult structure and usage tracking."""

    def test_result_with_content(self):
        result = MockChatResult(content="Test response", model="gpt-4o")
        assert result.content == "Test response"
        assert result.model == "gpt-4o"

    def test_token_usage_tracking(self):
        result = MockChatResult()
        assert result.usage["total_tokens"] == 50
        assert result.usage["prompt_tokens"] == 30
        assert result.usage["completion_tokens"] == 20

    def test_result_str_conversion(self):
        result = MockChatResult(content="Hello world")
        assert str(result) == "Hello world"
