"""模型抽象接口 — 支持多种 LLM 和 Embedding 提供者"""
from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: Optional[List[dict]] = None
    usage: dict = field(default_factory=dict)
    model: str = ""
    def __str__(self): return self.content


@dataclass
class StreamEvent:
    type: str
    content: str = ""
    tool_calls: Optional[List[dict]] = None


class BaseLLM(ABC):
    @abstractmethod
    async def chat(self, messages, tools=None, response_format=None, temperature=0.7, stream=False):
        pass
    @abstractmethod
    async def embed(self, texts):
        pass
    @abstractmethod
    async def embed_single(self, text):
        pass


class LLMFactory:
    @staticmethod
    def create(provider: str, model: str, **kwargs) -> BaseLLM:
        from mindforge.config import get_settings
        s = get_settings()
        if provider == "deepseek":
            from mindforge.models.deepseek_adapter import DeepSeekAdapter
            api_key = kwargs.pop("api_key", s.llm.deepseek_api_key) or s.llm.deepseek_api_key
            base_url = kwargs.pop("base_url", s.llm.deepseek_base_url) or s.llm.deepseek_base_url
            return DeepSeekAdapter(model=model, api_key=api_key, base_url=base_url)
        else:
            from mindforge.models.openai_adapter import OpenAIAdapter
            api_key = kwargs.pop("api_key", s.llm.openai_api_key) or s.llm.openai_api_key
            base_url = kwargs.pop("base_url", None) or s.llm.openai_base_url
            return OpenAIAdapter(model=model, api_key=api_key, base_url=base_url)
