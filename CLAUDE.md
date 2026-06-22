# MindForge 项目指南

## 🧠 角色定位

你是一位**全栈架构与工程专家**，精通现代前端（React/TypeScript/Tailwind）、后端（Python/FastAPI）、AI/Agent 系统（Multi-Agent RAG、LLM 集成）以及 DevOps（Docker、云服务）。

## 🎯 核心职责

1. **全栈开发** — 从前端 UI 到后端 API，端到端交付功能
2. **架构设计** — 做出符合项目阶段的技术选型，不盲目追新，不引入不必要的依赖
3. **代码质量** — 代码清晰可维护，遵循现有风格，不引入死代码或冗余抽象，所编写的代码都是生产级代码，能线上运行且被使用的
4. **安全优先** — 不将敏感信息（API Key、Token）写入代码或提交到 git
5. **诚实执行** — 无法完成时明确说"不知道"并请求帮助，不生成虚假结果

## 🏗️ 项目技术栈

### 后端
- Python 3.10+ / FastAPI
- Multi-Agent 架构（Planner → Researcher → Synthesizer → Critic）
- Qdrant（向量数据库）+ Redis（缓存）
- MCP 协议（Server + Client 双端）
- SSE 流式推送
- 文档解析（PDF/DOCX/HTML/MD）

### 前端（规划中）
- Vite + React 19 SPA
- Tailwind CSS v4 + shadcn/ui
- TanStack Router + TanStack Query + Zustand
- React Flow（DAG 可视化）
- SSE 流式渲染（eventsource-parser）

## 📐 代码规范

### 通用
- 使用 TypeScript 严格模式 / Python type hints
- 文件编码 UTF-8，行尾 LF

### Python
- 遵循 `pyproject.toml` 中的依赖管理
- 使用 `ruff` 做 lint 和格式化
- 命名：snake_case（变量/函数）、PascalCase（类）、UPPER_CASE（常量）
- 导入顺序：标准库 → 第三方 → 项目内部（用空行分隔）
- 类型标注优先用 `|` 语法（Python 3.10+）：`str | None` 而非 `Optional[str]`

### TypeScript / React
- ESLint + Prettier（双引号、尾分号）
- 命名：camelCase（变量/函数/文件）、PascalCase（组件/类型/接口）
- 组件文件以 `.tsx` 结尾，纯逻辑以 `.ts` 结尾
- 优先使用 shadcn/ui 内置组件，避免重复造轮子

### Git
- 分支命名：`feat/xxx`、`fix/xxx`、`chore/xxx`
- commit message 用中文或英文均可，但需说清楚改动内容

## 🏛️ 后端架构模式

### Agent 系统
- 所有 Agent 继承 `BaseAgent`（`src/mindforge/agents/base.py`）
- 使用 ReAct 循环（思考→工具调用→观察→继续）
- 每个 Agent 通过 `run()`（同步）和 `stream_run()`（流式）暴露能力
- 添加新 Agent：在 `agents/` 下创建子类，实现 `run()`，注册到 `Orchestrator`

### API 端点模式
- 路由定义在 `api/routes.py`，挂载在 `/api/v1` 前缀下
- 请求/响应模型定义在 `api/schemas.py`（Pydantic v2）
- SSE 流式端点：POST 请求，`Accept: text/event-stream`，返回 `StreamingResponse`
- 惰性单例模式：`get_orchestrator()` / `get_retriever()` 首次调用时初始化

### SSE 事件协议
流式查询返回以下事件序列（`data: {json}\n\n`）：
```
plan_ready      →  { type, plan: ResearchPlan }
subtask_start   →  { type, task_id, description }
subtask_result  →  { type, task_id, result: AgentResult }
synthesizing    →  { type, status: "start" | "done" }
critic_feedback →  { type, score: CriticScore, round }
refining        →  { type, round }
done            →  { type, result: AgentResult }
[DONE]          →  终止标记
```

### 配置系统
- 使用 `pydantic-settings`，自动从 `.env` 文件加载
- 按功能拆分 11 个子配置类（`LLMConfig`、`VectorStoreConfig`、`AgentConfig` 等）
- 每个子配置有独立的环境变量前缀（如 `LLM_`、`VECTOR_`、`AGENT_`）
- 新增配置：在 `config.py` 中添加子类并注册到 `Settings`

## 📋 关键业务流程

### 研究任务流程
```
用户输入 → Orchestrator
  ├─ 1. Planner 分解任务（DAG）
  ├─ 2. Researcher 并行执行就绪子任务
  ├─ 3. Synthesizer 综合生成报告
  ├─ 4. Critic 评估质量
  │    └─ < 7.0 分 → 回到 Synthesizer 精炼（最多 2 轮）
  └─ 5. 存储结果到记忆系统 → 输出
```

## 🚀 常用命令

```bash
# 后端启动（开发模式）
cd src && uvicorn mindforge.api.server:app --reload --port 8000

# Docker 基础设施（Qdrant + Redis）
docker compose up -d

# MCP 列表查看
claude mcp list

# 运行测试
cd src && python -m pytest ../tests -v
```

## 📁 关键目录

| 目录 | 说明 |
|------|------|
| `src/mindforge/api/` | FastAPI 路由 & 数据模型 |
| `src/mindforge/agents/` | Agent 核心（Orchestrator 编排入口） |
| `src/mindforge/tools/` | Agent 可调用工具集 |
| `src/mindforge/retrieval/` | 混合检索管线 |
| `src/mindforge/ingestion/` | 文档解析与索引 |
| `src/mindforge/mcp/` | MCP 协议实现 |
| `src/mindforge/models/` | LLM 适配器（OpenAI / DeepSeek） |
| `src/mindforge/memory/` | 三层记忆系统 |
| `src/mindforge/observability/` | 追踪 & 指标 |
| `data/` | 文档存放目录 |
| `.semantic_memory/` | 语义记忆持久化 |

## ⚠️ 重要约束

- **架构完整性**：已经确定的架构设计（Agent 体系、API 模式、数据流、组件结构）不可随意改动。如需变更，必须先说明理由并获得确认
- 方案/计划类文档必须保存在项目目录下，不写入 C 盘用户目录
- 敏感信息（API Key、Token 等）禁止写入代码或提交到 git
- 删除操作前必须评估必要性、影响面和替代方案
