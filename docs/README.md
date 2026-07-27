# MindForge 文档索引

本目录文档已于 2026-07-27 按当前 `main` 分支代码同步。运行参数以项目根目录 `.env.example` 为完整键清单，实际值以未提交的 `.env` 为准；源代码和自动化测试始终是行为事实的最终依据。

| 文档 | 用途 |
|------|------|
| [MindForge项目文档.md](MindForge项目文档.md) | 架构、模块、API、部署和分支管理 |
| [MindForge完整实现.md](MindForge完整实现.md) | 设计背景、关键实现和面试讲解材料 |
| [MindForge踩过的坑.md](MindForge踩过的坑.md) | 已复现问题、根因和修复方式 |
| [MindForge面试题目.md](MindForge面试题目.md) | Python、Agent、RAG、MCP、部署等题目 |
| [前端方案](../计划方案/MindForge前端方案.md) | 前端原方案、实际落地结果和后续约束 |

## 当前事实基线

- Python `>=3.10`，容器与 CI 使用 Python 3.11。
- React 19、TypeScript 6、Vite 8、Tailwind CSS 4。
- PostgreSQL-only，不提供 SQLite 回退。
- 根目录 `.env` 是运行和部署参数的唯一配置源。
- API 前缀为 `/api/v1`，当前包含 18 个 REST/JSON-RPC 路由。
- 研究流支持 `answer_chunk` 增量事件，默认最多精炼 1 轮。
- CI 执行 Ruff、74 项 pytest、前端 ESLint/构建和 Compose 配置校验。
- 生产部署使用 Docker Compose，FastAPI 单端口托管 API 与前端静态资源。
