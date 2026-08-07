# MindForge

MindForge 是一个可自托管的自适应 Agentic RAG 研究助理。它结合多 Agent 协作、
知识库检索、联网搜索、会话上下文、引用与可观测性，生成可追溯的研究结果。

## 核心能力

- 自适应请求路由：根据问题类型选择直接回复、直接回答或完整研究。
- 多 Agent 研究：Planner、Researcher、Synthesizer、Critic 分工完成规划、检索、汇总与评审。
- 会话上下文：支持连续追问、手动选择上下文、独立研究、上下文预览和运行快照。
- 知识库：支持 PDF、DOCX、HTML、Markdown、TXT 上传、解析、索引和检索。
- 混合检索：集成向量检索、BM25、RRF 融合、Cross-Encoder 重排序、RAPTOR 与 GraphRAG。
- 联网搜索：支持 Provider 原生搜索、Tavily 和 DuckDuckGo。
- 模型接入：支持 OpenAI、DeepSeek、Kimi、GLM、OpenAI-compatible API 及本地兼容服务。
- 实时交互：通过 SSE 展示路由、规划、检索、评审、精炼和回答生成进度，并支持取消任务。
- 可观测性：记录 Agent、模型、工具、耗时、Token、费用和失败原因；支持本地 Trace 与 Langfuse。

## 架构

```text
React 19 SPA
  |
  | REST / SSE
  v
FastAPI
  |
  +-- 请求路由与上下文服务
  +-- Multi-Agent Orchestrator
  +-- RAG / 联网搜索 / 代码执行 / 引用校验工具
  +-- Qdrant 混合检索与文档索引
  +-- PostgreSQL / Redis
  `-- Local Trace / Langfuse
```

## 研究流程

```text
用户问题
  -> 请求路由与上下文构建
  -> 直接回答或研究计划
  -> Researcher 检索与工具调用
  -> Synthesizer 汇总结果
  -> Critic 评审与精炼
  -> 引用校验、历史与 Trace 持久化
```

研究支持快速、均衡和深度三种模式，并可按配置使用知识库、联网搜索或自动来源策略。

## 上下文与记忆

研究工作台以 Conversation 组织连续问题。每轮任务可使用自动、手动、关闭和独立研究
四种上下文模式，并在执行前展示候选内容、来源、相关性和 Token 成本。

系统会保存不可变 `ContextSnapshot`，用于回看一次研究实际使用的历史内容。消息支持固定、
排除、遗忘和删除；有效研究产物与用户长期记忆可按相关性复用。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端 | Python 3.10+、FastAPI、Pydantic、SQLAlchemy、Alembic |
| 前端 | React 19、TypeScript、Vite、TanStack Router/Query、Zustand |
| 数据 | PostgreSQL、Qdrant、Redis |
| 检索 | BGE、BM25、Cross-Encoder Reranker、RAPTOR、GraphRAG |
| 文档解析 | pdfplumber、PaddleOCR、python-docx、BeautifulSoup |
| 观测 | 本地 Trace、Langfuse |

## 快速启动

推荐在 Linux 服务器使用 Docker Compose 部署。

### 1. 准备配置

```bash
git clone https://github.com/blankbrains/MindForge.git
cd MindForge
cp .env.example .env
```

在 `.env` 中配置数据库密码、`APP_SECRET`、LLM Provider、模型和 API Key。

### 2. 启动服务

CPU：

```bash
docker compose config --quiet
docker compose up -d --build
```

GPU：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up -d --build
```

### 3. 检查状态

```bash
curl --fail http://127.0.0.1:8000/api/v1/ready
docker compose ps
docker compose logs -f mindforge
```

服务默认地址为 `http://127.0.0.1:8000`，接口文档位于 `/docs`。

## 模型与搜索配置

设置页面支持为 Planner、Researcher、Critic、Synthesizer 和 Direct Answer 分别配置模型。
Provider 配置支持模型发现、工具调用、JSON 模式、流式用量统计和原生联网搜索能力。

联网搜索支持 Provider 原生搜索、Tavily 和 DuckDuckGo。搜索结果会携带来源信息，供报告引用、
评审和 Trace 使用。

## 知识库与检索

文档上传后以异步索引任务处理，前端展示进度并支持取消。PDF 解析支持原生文本、OCR、
表格、图片和版面元素处理。

检索管线结合 Dense、BM25、RRF 和 Reranker；RAPTOR 用于层次摘要索引，GraphRAG 用于
实体、关系和社区摘要检索。文档可查看内容、启停索引和删除。

## API 与可观测性

| API | 用途 |
| --- | --- |
| `POST /api/v1/query` | 提交普通或 SSE 流式研究任务 |
| `POST /api/v1/query/cancel` | 按 `request_id` 取消流式研究 |
| `POST /api/v1/index-jobs` | 创建异步文档索引任务 |
| `/api/v1/conversations/*` | 管理会话、消息和上下文 |
| `/api/v1/documents/*` | 管理知识库文档和资产 |
| `/api/v1/settings` | 管理模型、检索和运行配置 |
| `/api/v1/observability/*` | 查看和管理 Trace |
| `/api/v1/health` / `/api/v1/ready` | 检查服务健康与依赖就绪状态 |

每次研究会创建顶层 Trace，记录 Agent、模型、工具、耗时、Token、费用、结果与错误信息。
配置 Langfuse 后，同一链路会同步到外部观测平台。

## 配置分类

| 前缀 | 用途 |
| --- | --- |
| `API_*` | API、鉴权、上传限制和索引并发 |
| `LLM_*` | Provider、角色模型、Embedding 和费用 |
| `AGENT_*` | 研究模式、路由、超时和并发 |
| `WEB_SEARCH_*` / `TAVILY_*` | 联网搜索策略 |
| `RETRIEVAL_*` | 召回、重排序和阈值 |
| `PARSER_*` / `VISUAL_*` | 文档、OCR、表格和视觉解析 |
| `RAPTOR_*` / `GRAPH_*` | 层次索引和图谱索引 |
| `CONTEXT_*` / `MEMORY_*` | 会话上下文与记忆 |
| `OBSERVABILITY_*` | Trace 与 Langfuse |
| `SANDBOX_*` | 代码执行限制 |
| `VITE_*` | 前端请求和流式渲染 |

## 开发与验证

```bash
ruff check src/ tests/
pytest tests/ -v -m "not integration"

npm --prefix mindforge-web run lint
npm --prefix mindforge-web run test
npm --prefix mindforge-web run build

docker compose config --quiet
```

## 项目结构

```text
MindForge/
├── src/mindforge/
│   ├── agents/          # Agent 与 Orchestrator
│   ├── api/             # FastAPI、SSE 与业务接口
│   ├── context/         # 上下文、快照与删除治理
│   ├── ingestion/       # 文档解析、分块与索引
│   ├── retrieval/       # 向量、BM25、重排序与 GraphRAG
│   ├── models/          # LLM Provider 适配
│   ├── services/        # 会话、索引和健康检查服务
│   └── observability/   # Trace 与 Langfuse
├── mindforge-web/       # React 前端
├── migrations/          # Alembic 数据库迁移
├── tests/               # 后端测试
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
└── start.sh
```

## 安全说明

- `.env`、运行数据、索引、Trace、模型缓存和本地文档不进入 Git。
- API Key 加密保存，API 响应只返回脱敏值。
- 对外部署使用 HTTPS、访问控制和强随机密钥。
- 服务提供 CORS 配置、安全响应头、上传限制、模型发现保护和代码执行限制。

## License

[MIT](LICENSE)
