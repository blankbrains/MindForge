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
设置页从服务端读取运行时配置；浏览器只持久化非敏感参数和 Key 是否已配置的状态，完整 API Key 不写入 localStorage。

## 质量检查

```bash
npm test
npm run lint
npm run build
```

生产环境由根目录 `Dockerfile` 构建静态资源，再交给 FastAPI 单端口托管。
