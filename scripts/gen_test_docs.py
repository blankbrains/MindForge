#!/usr/bin/env python3
"""Generate professional test documents using DeepSeek API."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

os.environ["LLM_LLM_PROVIDER"] = "deepseek"
os.environ["LLM_DEEPSEEK_API_KEY"] = "sk-83763f65837f415193a260e74fd2bcf8"
os.environ["LLM_DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"

from mindforge.models.deepseek_adapter import DeepSeekAdapter
from mindforge.models.base import ChatMessage

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

llm = DeepSeekAdapter(model="deepseek-chat", api_key="sk-83763f65837f415193a260e74fd2bcf8")


async def gen_doc(filename, title, prompt):
    print(f"Generating: {filename}...")
    result = await llm.chat([
        ChatMessage(role="system", content="你是一个AI技术文档专家。用中文生成专业、详细、结构化的技术文档，包含标题、章节、代码示例和引用。"),
        ChatMessage(role="user", content=prompt),
    ], temperature=0.5, stream=False)

    content = result.content
    lines = content.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("# ") and i < 3:
            start = i + 1
            break

    full = f"# {title}\n\n" + "\n".join(lines[start:]).strip()

    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"  ✅ {filename} — {len(content)} chars")
    return filepath


async def main():
    topics = [
        ("rag_architecture.md", "RAG 架构深度解析",
         "生成一份关于 RAG（检索增强生成）架构的专业技术文档，要求：1. 解释 RAG 的核心原理和为什么需要 RAG 2. 详细描述 RAG 的完整流水线：文档解析→分块→向量化→索引→检索→生成 3. 对比 Naive RAG、Advanced RAG 和 Modular RAG 的区别 4. 讨论 RAG 的挑战：检索质量、上下文窗口、幻觉问题 5. 给出一个基于 Python 和 Qdrant 的简单 RAG 实现示例 6. 列举 RAG 的应用场景和最新研究趋势 要求专业、详实、不少于2000字。"),

        ("multi_agent_system.md", "Multi-Agent 系统设计与实践",
         "生成一份关于 Multi-Agent 系统的专业技术文档，要求：1. 解释什么是 Agent 以及 Agent 的核心能力（感知、推理、行动）2. 描述 Multi-Agent 架构模式：Orchestrator、Debate、Pipeline、Graph 3. 详细讨论 Agent 的工具使用和 ReAct 循环 4. 讨论 Agent 之间的通信和协作机制 5. 对比 LangGraph、AutoGen、CrewAI 等框架 6. 给出一个 Planner→Researcher→Critic→Synthesizer 的 Multi-Agent 实现示例 7. 讨论 Agent 系统的安全性和可观测性 要求专业、详实、不少于2000字。"),

        ("graphrag_and_raptor.md", "GraphRAG 与 RAPTOR：层次化与图结构检索",
         "生成一份关于 GraphRAG 和 RAPTOR 检索增强技术的专业技术文档，要求：1. 解释 GraphRAG 的核心思想：实体提取→关系构建→图社区发现→摘要 2. 解释 RAPTOR 的核心思想：文档分块→聚类→层次化摘要树 3. 对比两者的优缺点和适用场景 4. GraphRAG 适合跨文档关系发现，RAPTOR 适合单文档层次理解 5. 讨论如何将两者与传统的向量检索结合使用 6. 给出查询路由策略：根据不同查询类型选择不同检索策略 7. 讨论这些技术在 2024-2025 年的最新进展 要求专业、详实、不少于2000字。"),

        ("mcp_protocol.md", "MCP（Model Context Protocol）协议详解",
         "生成一份关于 MCP（Model Context Protocol）的专业技术文档，要求：1. 解释 MCP 是什么以及为什么需要它（工具调用的标准化问题）2. 描述 MCP 的架构：Host、Client、Server 三层模型 3. 详细解释 MCP 的 JSON-RPC 协议和生命周期 4. 讨论 MCP 的传输层：stdio 和 SSE 两种方式 5. 给出一个自定义 MCP Server 的实现示例 6. 对比 MCP 与其他工具调用方案（Function Calling、Plugin、Tool Protocol）7. 讨论 MCP 对 Agent 生态的影响和未来展望 要求专业、详实、不少于2000字。"),
    ]

    for filename, title, prompt in topics:
        try:
            await gen_doc(filename, title, prompt)
        except Exception as e:
            print(f"  ❌ {filename}: {e}")

    print("\n✅ 全部文档生成完成")
    for f in os.listdir(DOCS_DIR):
        fp = os.path.join(DOCS_DIR, f)
        size = os.path.getsize(fp)
        print(f"   {f}: {size} bytes")


if __name__ == "__main__":
    asyncio.run(main())
