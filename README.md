# MindForge

MindForge 是一个可自托管的自适应研究助理。它通过 Planner、Researcher、
Critic 和 Synthesizer 协作完成问题拆解、资料检索、证据整理、质量评审和报告生成，
并提供知识库、联网搜索、引用、历史记录与完整研究链路观测。

## 核心能力

- **自适应研究**：支持快速、均衡和深度模式，按问题复杂度选择直接回答或多任务研究。
- **多来源检索**：统一调度知识库、模型原生联网搜索和可选辅助搜索，来源不可用时明确降级。
- **文档知识库**：支持 PDF、DOCX、HTML、Markdown 和 TXT，覆盖 OCR、表格、图片与版面解析。
- **混合索引**：基础向量检索、BM25 和重排序，可按文档启用 RAPTOR 与 GraphRAG。
- **统一模型接入**：支持 OpenAI、DeepSeek、Kimi、GLM、通用 OpenAI-compatible API 和本地模型服务。
- **可追溯输出**：报告支持 Markdown、代码高亮、表格、可点击引用、质量评分和失败原因。
- **可观测链路**：每次研究建立顶层 Orchestrator Trace，展示 Agent、工具、模型、耗时、Token 和费用。

## 研究流程

```text
问题
  -> 研究模式与来源策略
  -> Planner 生成任务 DAG（简单问题可跳过）
  -> Researcher 并行执行检索与工具调用
  -> Synthesizer 组织报告
  -> Critic 评审并按需精炼
  -> 引用校验与结果持久化
```

| 模式 | 行为 |
| --- | --- |
| 快速 | 单任务执行，不进行质量精炼 |
| 均衡 | 简单问题快速处理，复杂问题自动规划 |
| 深度 | 完整 DAG、评审与精炼流程 |

来源策略支持自动选择、仅知识库和仅联网搜索。

## 技术栈

| 层级 | 主要技术 |
| --- | --- |
| 后端 | Python 3.10+、FastAPI、Pydantic、SQLAlchemy、Alembic |
| 前端 | React 19、TypeScript、Vite、TanStack Router/Query、Zustand |
| 数据 | PostgreSQL、Qdrant、Redis |
| 检索 | BGE、BM25、Cross-Encoder Reranker、RAPTOR、GraphRAG |
| 解析 | pdfplumber、PaddleOCR、python-docx、BeautifulSoup |
| 观测 | 本地 Trace、Langfuse（可选） |

## 快速启动

推荐在 Linux 服务器上使用 Docker Compose 部署。CPU 配置可以直接启动；GPU 配置需要
可用的 NVIDIA 驱动、设备文件和动态库。

### 1. 准备配置

```bash
git clone https://github.com/blankbrains/MindForge.git
cd MindForge
cp .env.example .env
```

至少检查以下配置：

- `POSTGRES_PASSWORD` 和与其一致的 `DATABASE_URL`
- `APP_SECRET`
- 使用的 LLM Provider、Base URL、API Key 和模型
- `DOCKER_API_BIND_ADDRESS`、`API_ACCESS_TOKEN` 等访问控制项

所有运行配置以项目根目录 `.env` 为准。完整字段和默认值见 `.env.example`。

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

MindForge 会同时启动应用、PostgreSQL、Qdrant 和 Redis，并在应用启动时自动执行
数据库迁移。

### 3. 检查状态

```bash
curl --fail http://127.0.0.1:8000/api/v1/ready
docker compose ps
docker compose logs -f mindforge
```

默认访问地址为 `http://127.0.0.1:8000`，FastAPI 接口文档位于 `/docs`。

停止服务：

```bash
docker compose down
```

命名数据卷默认保留。不要在未备份的情况下删除数据卷。

## 远程访问

生产环境应保持 `DOCKER_API_BIND_ADDRESS=127.0.0.1`，通过带 HTTPS 和身份认证的
反向代理开放应用，并由反向代理为 `/api/*` 注入
`Authorization: Bearer <API_ACCESS_TOKEN>`。

仅在完全受控的测试网络中直接访问服务器端口时，可以配置：

```dotenv
DOCKER_API_BIND_ADDRESS=0.0.0.0
API_ALLOW_INSECURE_REMOTE_ACCESS=true
```

测试结束后应恢复安全配置。PostgreSQL、Redis 和 Qdrant 端口不应暴露到公网。

## 模型配置

前端“设置”页面提供六类模型配置：

| Provider | 用途 |
| --- | --- |
| OpenAI | OpenAI 原生 API |
| DeepSeek | DeepSeek API |
| Kimi | Moonshot/Kimi API 与原生搜索 |
| GLM | 智谱 GLM API 与原生搜索 |
| 通用接口 | 其他 OpenAI-compatible 云服务 |
| 本地模型 | vLLM、Ollama、LM Studio 等兼容服务 |

填写 Base URL 和 API Key 后，可以从 Provider 的模型接口拉取模型列表，并分别为
Planner、Researcher、Critic 和 Synthesizer 选择模型。通用接口和本地服务需要正确
声明工具调用、JSON 模式及原生搜索能力。

设置页保存的 API Key 会加密存入 PostgreSQL，同时同步根目录 `.env`；接口返回时只
展示脱敏值。`LLM_MODEL_PRICING` 用于费用估算，未配置价格的模型会显示费用不可用，
不会错误显示为零费用。

## 联网搜索

联网能力按以下顺序组合：

1. Provider 原生搜索：OpenAI Responses、Kimi 内置搜索或 GLM Web Search。
2. Tavily：配置 `TAVILY_API_KEY` 后作为可选辅助搜索。
3. DuckDuckGo：通过 `WEB_SEARCH_DUCKDUCKGO_ENABLED` 显式启用。
4. 模型知识回答：无可用搜索来源时，可按配置降级，但不会伪造引用。

Tavily 不是必需项。支持原生搜索的模型可以直接联网；不支持原生搜索的模型需要配置
辅助搜索后才能返回可核验的外部来源。

## 知识库

支持上传：

- `.pdf`
- `.docx`
- `.html` / `.htm`
- `.md`
- `.txt`

上传任务异步执行，前端展示上传与索引进度，并支持取消。PDF 解析会按内容自动选择
原生文本提取或 OCR，同时处理表格、图片和版面元素。

基础索引包含向量、BM25 和重排序。上传时可选：

- **RAPTOR**：为长文档建立层次摘要索引。
- **GraphRAG**：抽取实体、关系和社区摘要。
- **视觉索引**：配置视觉模型后为图片资产生成可检索描述。

文档列表会显示实际应用的索引类型，并支持启用、停用、查看内容和删除文档。

## 可观测与历史

每次研究只创建一个顶层 Orchestrator Trace。本地观测页面提供：

- 任务状态、耗时和失败摘要
- Planner、Researcher、Critic、Synthesizer 执行链路
- 模型与工具调用详情
- Token 用量和费用估算
- Trace 搜索、筛选与删除

Langfuse 是可选的外部观测后端。配置
`OBSERVABILITY_LANGFUSE_PUBLIC_KEY`、`OBSERVABILITY_LANGFUSE_SECRET_KEY` 和
`OBSERVABILITY_LANGFUSE_HOST` 后，同一研究链路会同步到 Langfuse；未配置时本地
Trace 仍可使用。

研究历史和观测记录相互独立，均支持单条删除和清空。失败研究会保留可追踪的阶段、
错误类型和原因。

## 配置分类

| 配置前缀 | 作用 |
| --- | --- |
| `API_*` | 服务地址、鉴权、上传限制和索引并发 |
| `LLM_*` | Provider、模型、Embedding 和费用 |
| `AGENT_*` | 研究模式、超时、并发、任务和工具限制 |
| `WEB_SEARCH_*` / `TAVILY_*` | 联网搜索与降级策略 |
| `RETRIEVAL_*` | 混合检索、重排序和相关性阈值 |
| `PARSER_*` / `VISUAL_*` | 文档、OCR、表格、图片和视觉解析 |
| `RAPTOR_*` / `GRAPH_*` | 层次索引与图谱索引 |
| `OBSERVABILITY_*` | 本地 Trace 与 Langfuse |
| `VITE_*` | 前端请求、研究和流式渲染限制 |

修改 `.env` 后，影响进程初始化、模型加载或前端构建的配置需要重新构建并启动服务。

## 本地开发与验证

一体化启动脚本：

```bash
bash start.sh
bash start.sh --dev
```

质量检查与 CI 一致：

```bash
python3 -m pip install --require-hashes -r requirements-dev.lock
ruff check src/ tests/
pytest tests/ -v -m "not integration"

npm --prefix mindforge-web ci
npm --prefix mindforge-web run lint
npm --prefix mindforge-web run test
npm --prefix mindforge-web run build

cp .env.example .env
docker compose config --quiet
```

## 项目结构

```text
MindForge/
├── src/mindforge/
│   ├── agents/          # 四 Agent 与 Orchestrator
│   ├── api/             # FastAPI、SSE、设置与资源接口
│   ├── ingestion/       # 文档解析、分块、Embedding、RAPTOR
│   ├── retrieval/       # 向量、BM25、重排序、GraphRAG
│   ├── models/          # 模型与原生搜索适配器
│   ├── memory/          # 工作、情节与语义记忆
│   └── observability/   # 本地 Trace 与 Langfuse
├── mindforge-web/       # React 前端
├── migrations/         # Alembic 数据库迁移
├── tests/              # 后端测试
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
└── start.sh
```

## 安全说明

- `.env`、运行数据、索引、Trace、模型缓存和本地文档不进入 Git。
- 不要在代码、README、测试或脚本中写入真实 API Key、密码和服务器信息。
- 对外部署必须使用 HTTPS、访问控制和强随机密钥。
- 上传限制、解析限制、代码沙箱和并发限制均应按服务器资源调整。

## License

[MIT](LICENSE)
