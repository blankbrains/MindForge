# MindForge — 自适应研究助理系统

> Multi-Agent RAG + MCP 协议 + GraphRAG

## 项目概述

MindForge 是一个基于 Multi-Agent 架构的自适应研究助理系统。它能够接收用户提出的复杂研究问题，自动将问题分解为多步子任务，并行检索知识库和互联网信息，综合多源信息生成结构化的研究报告，并通过自我批评机制迭代优化输出质量。

### 核心能力

| 能力 | 说明 |
|------|------|
| **智能任务分解** | Planner Agent 将复杂问题拆解为 DAG 子任务，自动识别依赖关系 |
| **多源信息检索** | 同时检索内部知识库（Qdrant 向量库）和互联网实时信息 |
| **自适应检索策略** | 根据问题类型（事实/概念/比较/流程/分析/关系）自动选择最优检索策略 |
| **自我批评优化** | Critic Agent 从 5 个维度评分，低于阈值自动触发精炼循环 |
| **标准化工具接入** | 通过 MCP 协议动态发现和调用外部工具，支持热插拔 |
| **OpenAI / DeepSeek 双引擎** | 模型层抽象化，一键切换，DeepSeek 成本降低 90% |
| **流式输出** | 支持 SSE 流式推送，前端可实时展示 Agent 思考过程和中间结果 |

### 工作流程

```
用户输入问题
    ↓
┌─ Planner Agent ─────────────────────────────┐
│  任务分解：将问题拆解为 DAG 子任务            │
│  识别依赖：分析子任务间的先后关系              │
└──────────────────┬─────────────────────────┘
                   ↓
┌─ Researcher Agent ──────────────────────────┐
│  并行执行子任务                               │
│  ├── RAGTool（知识库检索）                    │
│  ├── WebSearchTool（互联网搜索）              │
│  ├── CodeExecutor（代码执行/数据分析）         │
│  ├── CitationVerifier（引用验证）             │
│  └── MCPToolAdapter（外部 MCP 工具）          │
└──────────────────┬─────────────────────────┘
                   ↓
┌─ Synthesizer Agent ─────────────────────────┐
│  综合所有子任务结果                            │
│  生成结构化研究报告（摘要→分析→结论→引用）      │
└──────────────────┬─────────────────────────┘
                   ↓
┌─ Critic Agent ──────────────────────────────┐
│  5 维度评分（完整性/准确性/深度/清晰度/引用质量）│
│  < 7.0 分 → 返回 Synthesizer 精炼            │
│  ≥ 7.0 分 → 输出最终报告                     │
└─────────────────────────────────────────────┘
```

### 技术栈

| 层 | 技术 |
|---|------|
| Agent 框架 | LangGraph Multi-Agent (Planner→Researcher→Critic→Synthesizer) |
| 检索引擎 | Qdrant（向量库）+ BM25（稀疏检索）+ RRF 融合 + CrossEncoder 精排 |
| 层次化检索 | RAPTOR Tree（自底向上摘要树） |
| 图谱检索 | GraphRAG（跨文档实体关系发现） |
| 工具协议 | MCP (Model Context Protocol) — 标准化工具接入 |
| 模型 | OpenAI GPT-4o / DeepSeek-chat 一键切换 |
| 记忆系统 | 工作记忆 + 情节记忆 + 语义记忆 三层架构 |
| 质量保障 | Critic Agent 自评分 + Self-Refine 精炼循环 |
| 防幻觉 | 引用验证工具 + 多源交叉验证 |
| 服务 | FastAPI + SSE 流式输出 |
| 可观测 | LangFuse + 本地 JSONL 追踪 |
| 部署 | Docker Compose（Qdrant + Redis + API） |

## 项目结构

```
mindforge/
├── pyproject.toml                 # 项目依赖管理
├── docker-compose.yml             # Docker 编排（Qdrant + Redis + API）
├── Dockerfile                     # 容器构建
├── .env.example                   # 环境变量模板
│
├── scripts/
│   ├── run_research.py            # 研究任务执行脚本
│   └── mcp_discover.py            # MCP 工具发现脚本
│
├── src/mindforge/
│   ├── config.py                  # 统一配置管理（Pydantic Settings）
│   │
│   ├── ingestion/                 # 文档处理流水线
│   │   ├── parsers.py             # 多格式解析（PDF/DOCX/HTML/MD/TXT）
│   │   ├── chunker.py             # 文本分块（递归分割 + 语义分割）
│   │   ├── embedder.py            # Embedding 生成（OpenAI / BGE 双模式）
│   │   └── raptor.py              # RAPTOR 层次化索引
│   │
│   ├── retrieval/                 # 检索系统
│   │   ├── vector_store.py        # Qdrant 向量库封装
│   │   ├── bm25.py                # BM25 稀疏检索
│   │   ├── hybrid.py              # 混合检索 + RRF 融合
│   │   ├── reranker.py            # CrossEncoder 精排
│   │   ├── adaptive.py            # 自适应检索策略路由
│   │   └── graphrag.py            # GraphRAG 引擎
│   │
│   ├── agents/                    # Multi-Agent 系统
│   │   ├── base.py                # Agent 基类（工具循环框架）
│   │   ├── planner.py             # Planner Agent（DAG 任务分解）
│   │   ├── researcher.py          # Researcher Agent（ReAct 研究执行）
│   │   ├── critic.py              # Critic Agent（5 维质量评估）
│   │   ├── synthesizer.py         # Synthesizer Agent（报告生成）
│   │   └── orchestrator.py        # 编排器（多 Agent 调度）
│   │
│   ├── memory/                    # 记忆系统
│   │   ├── working.py             # 工作记忆（任务内上下文）
│   │   ├── episodic.py            # 情节记忆（跨会话历史）
│   │   └── semantic.py            # 语义记忆（持久化事实）
│   │
│   ├── tools/                     # Agent 工具
│   │   ├── base.py                # 工具基类
│   │   ├── rag_tool.py            # 知识库检索工具
│   │   ├── web_search.py          # 网络搜索工具
│   │   ├── code_executor.py       # 代码执行工具
│   │   ├── citation_verifier.py   # 引用验证工具
│   │   └── mcp_adapter.py         # MCP 协议适配器
│   │
│   ├── mcp/                       # MCP 协议层
│   │   ├── registry.py            # MCP 工具注册表
│   │   ├── client.py              # MCP 客户端（调用外部工具）
│   │   └── server.py              # MCP 服务端（暴露 Agent 能力）
│   │
│   ├── models/                    # 模型层
│   │   ├── base.py                # 模型抽象接口
│   │   ├── openai_adapter.py      # OpenAI 适配器
│   │   └── deepseek_adapter.py    # DeepSeek 适配器
│   │
│   ├── observability/             # 可观测性
│   │   ├── tracer.py              # 链路追踪（LangFuse + 本地）
│   │   └── metrics.py             # 指标收集
│   │
│   └── api/                       # API 服务层
│       ├── schemas.py             # 请求/响应模型
│       ├── routes.py              # REST 路由
│       └── server.py              # FastAPI 应用
│
└── tests/
    ├── test_retrieval.py
    ├── test_mcp_adapter.py
    └── test_models.py
```

### 架构分层

```
┌──────────────────────────────────────────────────────────┐
│                      API 服务层                           │
│        FastAPI + SSE 流式 + LangFuse 可观测性              │
├──────────────────────────────────────────────────────────┤
│                   Agent 推理层                             │
│   Planner → Researcher → Synthesizer → Critic（循环精炼）  │
├──────────────────────────────────────────────────────────┤
│                    工具层                                  │
│   RAG / Web Search / Code Exec / Citation / MCP Adapter   │
├──────────────────────────────────────────────────────────┤
│                    检索层                                  │
│   Qdrant + BM25 + RAPTOR + GraphRAG + CrossEncoder        │
├──────────────────────────────────────────────────────────┤
│                  模型 + 记忆 + MCP                          │
│   OpenAI/DeepSeek  │  三层记忆  │  MCP 协议生态             │
└──────────────────────────────────────────────────────────┘
```

## 开始使用

### 环境要求

- Python 3.10+
- Docker（可选，用于 Qdrant 和 Redis）

### 快速启动（本地开发）

```bash
# 1. 克隆项目
git clone <repo-url> && cd mindforge

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，设置 LLM_PROVIDER=deepseek 或 openai，并填入对应 API Key

# 5. 启动基础设施（需要 Docker）
docker-compose up -d qdrant redis

# 6. 验证 API 连通性
python scripts/run_research.py

# 7. 启动 API 服务
uvicorn mindforge.api.server:app --reload --port 8000

# 8. 访问 API 文档
open http://localhost:8000/docs
```

### 快速启动（无 Docker）

```bash
# 不启动 Qdrant/Redis，仅测试 API 和 Agent 核心逻辑
python scripts/run_research.py
uvicorn mindforge.api.server:app --reload --port 8000
# 注意：无 Qdrant 时检索功能不可用，但 Agent 推理和 LLM 调用可正常运行
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/query` | 提交研究任务（支持 sync / SSE stream） |
| POST | `/api/v1/index` | 索引文档到知识库 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/stats` | 系统统计 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| GET | `/` | 服务信息 |

### 调用示例

```bash
# 提交研究任务
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "解释Transformer中的自注意力机制",
    "stream": false
  }'

# 流式调用
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "比较RAG和微调两种方法的优劣",
    "stream": true
  }'

# 索引文档
curl -X POST http://localhost:8000/api/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "file_path": "data/docs/transformer_intro.md",
    "use_raptor": true,
    "use_graphrag": true
  }'
```

## 技术亮点

| 特性 | 说明 |
|------|------|
| Multi-Agent 流水线 | Planner→Researcher→Synthesizer→Critic，职责单一、可独立优化 |
| DAG 任务分解 | 复杂问题自动拆解为有向无环图，支持并行执行 |
| MCP 协议集成 | 工具通过 Model Context Protocol 标准化接入，支持动态发现 |
| 双引擎检索 | RAPTOR 层次化索引 + GraphRAG 实体图谱，适配不同查询类型 |
| 自适应策略 | 根据 6 种查询类型自动路由到最优检索策略 |
| 多模型支持 | OpenAI / DeepSeek 一键切换，对上层 Agent 完全透明 |
| 自我批评 | Critic Agent 5 维度评分 + Self-Refine 迭代精炼 |
| 三层记忆 | 工作记忆（任务内）+ 情节记忆（跨会话）+ 语义记忆（持久化） |
| 引用验证 | 自动验证报告中的引用是否能在源文档中找到 |
| 全链路可观测 | LangFuse + 本地 JSONL 双写追踪 |
