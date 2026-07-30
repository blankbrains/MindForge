# MindForge Web

MindForge 的 React 19 单页前端，负责研究任务流式交互、知识库管理、历史记录和运行时设置。

## 技术栈

- React 19 + TypeScript 6
- Vite 8 + Tailwind CSS 4
- TanStack Router + TanStack Query
- Zustand
- React Flow + Recharts
- `eventsource-parser` + `react-markdown`

## 本地开发

前端和后端、Docker Compose 共用项目根目录 `.env`。Vite 通过 `envDir` 读取其中的 `VITE_*` 配置。

```bash
cd mindforge-web
npm ci
npm run dev
```

默认地址由根目录 `.env` 中的 `VITE_DEV_HOST` 和 `VITE_DEV_PORT` 控制，`/api` 请求代理到 `VITE_API_PROXY_TARGET`。
设置页从服务端读取运行时配置；保存后必须重新读取成功才会显示“已保存”。浏览器只持久化非敏感参数和 Key 是否已配置的状态，完整 API Key 不写入 localStorage。
模型路由可按当前 Base URL 和凭证拉取服务端 `/models` 列表，四个 Agent
可直接选择返回的模型；未枚举的模型仍可通过自定义模型 ID 输入。

研究结果和历史详情共用 Markdown、GFM、代码高亮与结构化来源渲染。正文 `[N]`
会安全跳转到外部 `http/https` 来源或报告底部的内部知识库来源，代码块和已有链接
不会被二次改写。知识库文档卡片展示后端记录的实际索引能力，不把上传时勾选但被
跳过的 RAPTOR/GraphRAG 显示为已启用。

## 质量检查

```bash
npm test
npm run lint
npm run build
```

生产环境由根目录 `Dockerfile` 构建静态资源，再交给 FastAPI 单端口托管。
