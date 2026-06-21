#!/usr/bin/env python3
"""MindForge 研究任务执行脚本"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("MindForge — 自适应研究助理系统")
    print("=" * 60)

    # Step 1: Check configuration
    print("\n[1/4] 检查配置...")
    from mindforge.config import get_settings
    settings = get_settings()
    print(f"   模型提供商: {settings.llm.llm_provider}")
    print(f"   API Key 就绪: {bool(getattr(settings.llm, f'{settings.llm.llm_provider}_api_key', ''))}")

    # Step 2: Prepare knowledge base document
    print("\n[2/4] 准备知识库文档...")
    doc_dir = Path("data/docs")
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "transformer_intro.md"
    doc_path.write_text("""# Transformer 架构简介

## 自注意力机制
自注意力机制（Self-Attention）是 Transformer 的核心创新。
它允许模型在处理序列时，直接捕捉任意两个位置之间的关系。
计算公式：Attention(Q, K, V) = softmax(QK^T / d_k) V

## 多头注意力
多头注意力（Multi-Head Attention）通过并行计算多组注意力，
让模型从不同表示子空间学习信息。

## 位置编码
由于自注意力本身没有位置感知能力，
Transformer 使用正弦位置编码来注入位置信息。

## 前馈网络
每个 Transformer 层包含一个前馈神经网络（FFN），
对每个位置的表示进行非线性变换。
""", encoding="utf-8")
    print(f"   文档已创建: {doc_path.name}")

    from mindforge.ingestion.parsers import DocumentParser
    from mindforge.ingestion.chunker import TextSplitter
    parser = DocumentParser()
    doc = parser.parse(str(doc_path))
    splitter = TextSplitter()
    chunks = splitter.split(doc.doc_id, doc.content)
    print(f"   解析完成: {len(doc.content)} 字符, {len(chunks)} 个块")

    # Step 3: Test DeepSeek API
    print("\n[3/4] 测试 LLM API...")
    try:
        from mindforge.models.base import ChatMessage
        if settings.llm.llm_provider == "deepseek":
            from mindforge.models.deepseek_adapter import DeepSeekAdapter
            llm = DeepSeekAdapter(
                model=settings.llm.get_model("researcher"),
                api_key=settings.llm.deepseek_api_key,
                base_url=settings.llm.deepseek_base_url,
            )
        else:
            from mindforge.models.openai_adapter import OpenAIAdapter
            llm = OpenAIAdapter(
                model=settings.llm.get_model("researcher"),
                api_key=settings.llm.openai_api_key,
                base_url=settings.llm.openai_base_url,
            )

        result = await llm.chat([
            ChatMessage(role="system", content="你是一个AI专家，用中文简洁回答。"),
            ChatMessage(role="user", content="请用一句话解释Transformer中的自注意力机制是什么？"),
        ], temperature=0.3)

        print(f"   [{settings.llm.llm_provider} Response]")
        print(f"   {result.content[:200]}")
        print(f"   Token用量: {result.usage.get('total_tokens', 'N/A')}")
        print(f"   API 调用成功!")

    except Exception as e:
        print(f"   API 调用失败: {e}")
        print(f"   请检查 API Key 配置")

    # Step 4: Start API server info
    print("\n[4/4] 服务就绪")
    print()
    print(f"   API 服务: uvicorn mindforge.api.server:app --reload --port 8000")
    print(f"   API 文档: http://localhost:8000/docs")
    print(f"   健康检查: http://localhost:8000/api/v1/health")
    print()
    print("=" * 60)
    print("系统就绪!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
