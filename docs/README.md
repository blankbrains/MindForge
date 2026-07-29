# MindForge 文档索引

本目录文档已于 2026-07-29 按当前 `main` 分支代码和自动化测试结果同步。运行参数以项目根目录 `.env.example` 为完整键清单，实际值以未提交的 `.env` 为准；源代码和自动化测试始终是行为事实的最终依据。

| 文档 | 用途 |
|------|------|
| [MindForge项目文档.md](MindForge项目文档.md) | 架构、模块、API、部署和分支管理 |
| [MindForge完整实现.md](MindForge完整实现.md) | 设计背景、关键实现和面试讲解材料 |
| [MindForge踩过的坑.md](MindForge踩过的坑.md) | 已复现问题、根因和修复方式 |
| [MindForge面试题目.md](MindForge面试题目.md) | Python、Agent、RAG、MCP、部署等题目 |
| [自适应文档解析管线.md](自适应文档解析管线.md) | PDF 原生文本、OCR、表格、图片元素和索引进度 |
| [document-parsing-operations.md](document-parsing-operations.md) | 文档资产、视觉检索、版本治理和私有基准运维 |
| [前端方案](../计划方案/MindForge前端方案.md) | 前端原方案、实际落地结果和后续约束 |

## 当前事实基线

- Python `>=3.10`，容器与 CI 使用 Python 3.11。
- React 19、TypeScript 6、Vite 8、Tailwind CSS 4。
- PostgreSQL-only，不提供 SQLite 回退。
- 根目录 `.env` 是运行和部署参数的唯一配置源。
- `DATABASE_URL` 为后端启动必填项；应用不提供内置数据库连接串回退。
- API 前缀为 `/api/v1`，当前包含 22 个 REST/SSE 路由方法。
- 当前 Web 应用已停用 MCP；旧源码、脚本和测试已移除，协议说明仅作历史学习参考。
- 研究流支持 `planning`、`heartbeat` 和 `answer_chunk` 事件，默认最多精炼 1 轮；未配置 LLM Key 时直接进入知识库检索模式。
- RAPTOR 跳过单节点摘要并批量生成摘要向量；GraphRAG 可由 Agent 的 `auto/graph` 模式触发，使用构建快照与社区摘要复用。
- 当前验证基线包含 143 项 pytest、20 项前端回归测试、ESLint、构建和 Compose 校验。
- 模型层通过 Provider Registry 统一接入 OpenAI、DeepSeek、兼容云 API 与本地
  推理服务；设置页可独立配置 Base URL、Key、角色模型和 Tool/JSON 能力。
- 数据库通过 Alembic 迁移，当前迁移头为 `0005_document_assets`。
- CPU 与 GPU 使用互斥依赖锁；服务器当前运行 `torch==2.13.0+cu130`，
  `torch.cuda.is_available()` 为 `True`，设备为 NVIDIA GPU。
- PDF 默认最多 600 页；大文档使用有界并发解析、持久化索引任务和内容签名复用。
- 目标环境没有 NVIDIA Container Toolkit，GPU Compose 通过显式映射设备节点和
  驱动库运行。Reranker 模型未缓存且容器无外网，启动熔断后使用
  向量 + BM25 + RRF，不声称 CrossEncoder 已启用。
- 知识库页面不展示历史索引任务；上传弹窗使用 XHR 显示真实字节传输进度，
  上传完成后只轮询当前索引任务。
- PDF 解析使用自适应按页管线：原生文本优先，低质量页再使用 PaddleOCR；
  原生表格转 Markdown，OCR/表格/图片元素携带页码、坐标、置信度与来源方法。
  资产、解析版本和页级指标会持久化；视觉描述检索默认关闭，显式配置后才会执行。
  `PARSER_*` 为完整运行配置，默认使用可访问的 Paddle BOS 模型源。
- 生产部署使用 Docker Compose，FastAPI 单端口托管 API 与前端静态资源。
