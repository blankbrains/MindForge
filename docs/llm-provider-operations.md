# LLM Provider 配置与运维

MindForge 的 Agent、RAPTOR、GraphRAG 和 QA 生成统一通过 `LLMFactory` 调用模型。
当前内置四种 Provider：

| Provider | 用途 | API Key | Base URL | 模型配置 |
|----------|------|---------|----------|----------|
| `openai` | OpenAI 原生 API | 必填 | 默认 `https://api.openai.com/v1` | 四个角色模型 |
| `deepseek` | DeepSeek 原生 API | 必填 | 默认 `https://api.deepseek.com` | 四个角色模型 |
| `openai_compatible` | 兼容 OpenAI Chat Completions 的云端 API | 默认必填 | 必填 | 默认模型 + 可选角色覆盖 |
| `local` | 服务器上的 vLLM、Ollama、LM Studio 等服务 | 默认可选 | 必填 | 默认模型 + 可选角色覆盖 |

## 一、通过设置页配置

### 1. 打开设置页

生产部署默认通过应用端口访问：

```text
http://<服务器地址>:<API_PORT>/settings
```

进入“LLM 供应商”页签后，选择需要启用的 Provider。

### 2. 填写连接参数

#### OpenAI

| 字段 | 填写方式 |
|------|----------|
| Base URL | 默认 `https://api.openai.com/v1`；代理网关填写其完整 HTTP(S) 地址 |
| API Key | 填写 OpenAI Key |
| Planner / Researcher / Critic / Synthesizer | 填写账号实际可用的模型 ID |

#### DeepSeek

| 字段 | 填写方式 |
|------|----------|
| Base URL | 保持 `https://api.deepseek.com`，除非使用可信代理 |
| API Key | 填写 DeepSeek Key |
| 四个角色模型 | 通常可统一填写 `deepseek-chat`，也可按角色分别配置 |

#### OpenAI 兼容云 API

| 字段 | 填写方式 |
|------|----------|
| Base URL | 云服务提供的 OpenAI-compatible 根地址，通常以 `/v1` 结尾 |
| API Key | 填写该云服务的 Key；无鉴权服务可关闭“需要 API Key” |
| 默认模型 | 填写服务返回的准确模型 ID |
| 角色模型 | 留空时继承默认模型；需要分流时再单独填写 |

Base URL 必须是绝对 HTTP(S) URL，不能包含用户名、密码、查询参数或 fragment。

#### 本地模型

MindForge 容器通过宿主机网关访问本地推理服务。常用 Base URL：

```text
Ollama: http://host.docker.internal:11434/v1
vLLM:   http://host.docker.internal:8001/v1
```

如果 MindForge 后端不是运行在 Docker 中，可改用：

```text
http://127.0.0.1:<模型服务端口>/v1
```

本地 Provider 默认关闭“需要 API Key”。若 vLLM 等服务启用了鉴权，则打开该开关
并填写对应 Key。

### 3. 拉取模型列表

填写 Base URL 和 API Key 后点击“拉取模型”。后端会使用当前表单草稿请求该
Base URL 下的标准 `/models` 端点，去重后把模型 ID 返回设置页：

- OpenAI、DeepSeek、OpenAI-compatible 云 API 和提供 `/v1/models` 的本地服务
  使用同一发现协议。
- 已保存的脱敏 Key 由后端解密使用，明文不会返回浏览器。
- 云 Provider 的模型发现禁止访问私网、链路本地和保留地址；本地 Provider
  可访问回环和私网模型服务，但仍禁止链路本地元数据地址。
- 不跟随上游重定向，响应大小、超时和最大模型数量由 `API_MODEL_DISCOVERY_*`
  配置限制。
- 接口没有 `/models` 或目标模型未出现在列表时，选择“自定义模型 ID”并手动输入。

连接参数改变后，旧模型列表会立即失效；尚未完成的旧请求也不能覆盖新配置。

### 4. 配置模型路由

四个角色的用途如下：

| 角色 | 用途 | 建议 |
|------|------|------|
| Planner | 拆解任务、生成 DAG | 需要稳定结构化输出 |
| Researcher | 检索与工具调用 | 需要 Tool Calling |
| Critic | 质量评分与反馈 | 需要稳定 JSON 输出 |
| Synthesizer | 汇总并生成报告 | 优先选择上下文较长的模型 |

`openai_compatible` 和 `local` 可以只填写“默认模型”，四个角色留空时会继承默认
模型。OpenAI 和 DeepSeek 使用各自的角色模型字段。

### 5. 设置接口能力

兼容云 API 和本地模型需要按真实能力配置：

| 开关 | 开启条件 | 关闭后的行为 |
|------|----------|--------------|
| 需要 API Key | 服务端启用了鉴权 | 允许无 Key 初始化 |
| 工具调用 | 模型和推理服务均支持 Tool Calling | 不向接口发送 `tools` |
| JSON Mode | 接口支持 `json_object` | 不发送 JSON Mode 参数 |
| JSON Schema | 接口支持 `json_schema` | 不发送 JSON Schema 参数 |
| 流式 usage | 流式响应支持 `stream_options.include_usage` | 流式综合阶段不统计 Token |

不确定时先关闭 Tool Calling、JSON Mode 和 JSON Schema，确认普通 Chat 可用后逐项
启用。关闭 Tool Calling 后模型仍可生成文本，但 Researcher 无法完成完整工具循环。

### 6. 保存并验证

点击“保存配置”。保存成功后：

1. 当前 Provider 右上角应显示“可用”。
2. 研究页不再显示“文档检索模式”提示。
3. 可调用设置 API 检查状态；API Key 只返回脱敏值。

```bash
curl -s http://127.0.0.1:8000/api/v1/settings
curl -s http://127.0.0.1:8000/api/v1/ready
```

`llm_configured=true` 表示当前 Provider 的 Base URL、模型和 Key 要求均已满足。

## 二、部署本地推理服务

MindForge 只连接推理服务，不在应用容器中直接加载生成式大模型。

### 1. Ollama 示例

在 MindForge 所在服务器宿主机执行：

```bash
ollama pull qwen3:8b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
curl http://127.0.0.1:11434/v1/models
```

设置页填写：

```text
Provider: 本地模型
Base URL: http://host.docker.internal:11434/v1
默认模型: qwen3:8b
需要 API Key: 关闭
```

模型 ID 必须以 `/v1/models` 的实际返回值为准。

### 2. vLLM 示例

MindForge 默认使用 8000 端口，因此示例让 vLLM 使用 8001：

```bash
vllm serve <模型仓库或本地路径> \
  --host 0.0.0.0 \
  --port 8001

curl http://127.0.0.1:8001/v1/models
```

如需鉴权，可在启动时增加 `--api-key <your-key>`，或设置 `VLLM_API_KEY`。此时
MindForge 的“需要 API Key”必须开启并保存同一 Key：

```bash
curl -H "Authorization: Bearer <your-key>" \
  http://127.0.0.1:8001/v1/models
```

设置页填写：

```text
Provider: 本地模型
Base URL: http://host.docker.internal:8001/v1
默认模型: <vLLM /v1/models 返回的模型 ID>
需要 API Key: 按 vLLM 启动参数设置
```

Tool Calling 是否可用取决于模型 Chat Template 和 vLLM 的工具解析配置，不能只根据
接口地址判断。

### 3. 容器连通性检查

```bash
docker compose exec mindforge getent hosts host.docker.internal
docker compose exec mindforge \
  curl -s http://host.docker.internal:11434/v1/models
```

如果宿主机访问正常而容器访问失败，检查：

1. 模型服务是否监听 `0.0.0.0`，而不是只监听 `127.0.0.1`。
2. 模型服务端口是否被防火墙拦截。
3. `docker-compose.yml` 是否包含
   `host.docker.internal:host-gateway`。

## 三、通过 `.env` 配置

设置页最终也会把运行配置写入项目根目录 `.env`。首次部署或自动化部署可直接按键
配置，但不能用本地 `.env` 整文件覆盖服务器 `.env`。

### OpenAI

```dotenv
LLM_LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=<your-key>
LLM_OPENAI_BASE_URL=https://api.openai.com/v1
LLM_PLANNER_MODEL=gpt-4o
LLM_RESEARCHER_MODEL=gpt-4o-mini
LLM_CRITIC_MODEL=gpt-4o
LLM_SYNTHESIZER_MODEL=gpt-4o
```

### DeepSeek

```dotenv
LLM_LLM_PROVIDER=deepseek
LLM_DEEPSEEK_API_KEY=<your-key>
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_DEEPSEEK_PLANNER=deepseek-chat
LLM_DEEPSEEK_RESEARCHER=deepseek-chat
LLM_DEEPSEEK_CRITIC=deepseek-chat
LLM_DEEPSEEK_SYNTHESIZER=deepseek-chat
```

### OpenAI 兼容云 API

```dotenv
LLM_LLM_PROVIDER=openai_compatible
LLM_COMPATIBLE_API_KEY=<your-key>
LLM_COMPATIBLE_BASE_URL=https://<provider-host>/v1
LLM_COMPATIBLE_API_KEY_REQUIRED=true
LLM_COMPATIBLE_MODEL=<model-id>
LLM_COMPATIBLE_PLANNER_MODEL=
LLM_COMPATIBLE_RESEARCHER_MODEL=
LLM_COMPATIBLE_CRITIC_MODEL=
LLM_COMPATIBLE_SYNTHESIZER_MODEL=
LLM_COMPATIBLE_SUPPORTS_TOOLS=true
LLM_COMPATIBLE_SUPPORTS_JSON_MODE=true
LLM_COMPATIBLE_SUPPORTS_JSON_SCHEMA=false
LLM_COMPATIBLE_SUPPORTS_STREAM_USAGE=false
```

### 本地模型

```dotenv
LLM_LLM_PROVIDER=local
LLM_LOCAL_API_KEY=
LLM_LOCAL_BASE_URL=http://host.docker.internal:11434/v1
LLM_LOCAL_API_KEY_REQUIRED=false
LLM_LOCAL_MODEL=qwen3:8b
LLM_LOCAL_PLANNER_MODEL=
LLM_LOCAL_RESEARCHER_MODEL=
LLM_LOCAL_CRITIC_MODEL=
LLM_LOCAL_SYNTHESIZER_MODEL=
LLM_LOCAL_SUPPORTS_TOOLS=true
LLM_LOCAL_SUPPORTS_JSON_MODE=true
LLM_LOCAL_SUPPORTS_JSON_SCHEMA=false
LLM_LOCAL_SUPPORTS_STREAM_USAGE=false
```

修改后执行：

```bash
docker compose config --quiet
docker compose up -d --build mindforge
curl --fail http://127.0.0.1:8000/api/v1/ready
```

模型发现资源边界：

```dotenv
API_MODEL_DISCOVERY_TIMEOUT_SECONDS=15
API_MODEL_DISCOVERY_MAX_RESPONSE_BYTES=2097152
API_MODEL_DISCOVERY_MAX_MODELS=1000
```

## 四、Token 与估算费用

Provider 的 Chat Completions 响应只提供 Token 用量，MindForge 根据 `.env` 中的
模型单价计算估算费用：

```dotenv
LLM_MODEL_PRICING={"provider:model-id":{"input":0.0,"cached_input":0.0,"output":0.0}}
```

- 单价单位是 USD / 100 万 Token，必须替换为供应商当前公布的实际价格。
- 键优先使用 `provider:model-id`，也支持只写模型 ID。
- `cached_input` 可省略；省略时按普通输入价格估算。
- OpenAI 和 DeepSeek 原生流式接口会请求 usage；兼容云和本地端点默认关闭，
  只有确认支持时才开启 `LLM_COMPATIBLE_SUPPORTS_STREAM_USAGE` 或
  `LLM_LOCAL_SUPPORTS_STREAM_USAGE`。
- 页面展示“估算费用”而不是实际账单，并区分“未配置模型价格”“API 未返回用量”
  “不涉及 API 费用”和“部分估算”。

模型切换后必须同步更新 `LLM_MODEL_PRICING`，否则系统会明确显示未配置价格，不会
用其他模型的价格猜测，也不会显示为 `$0`。

## 五、高级索引和脚本模型

以下配置留空时会继承当前 Provider 的 Researcher 模型：

```dotenv
RAPTOR_SUMMARY_MODEL=
GRAPH_ENTITY_EXTRACTION_MODEL=
GRAPH_COMMUNITY_SUMMARY_MODEL=
QA_MODEL=
```

只有明确需要不同模型时才填写覆盖值，而且该模型 ID 必须属于当前 Provider。

辅助脚本也使用统一 Provider：

```bash
python scripts/run_research.py
python scripts/gen_test_docs.py
python scripts/generate_qa_dataset.py --domain computer_science
```

## 六、常见问题

### 保存后仍显示“未就绪”

检查当前 Provider：

- 模型名是否为空。
- 非 OpenAI Provider 的 Base URL 是否为空。
- “需要 API Key”开启时是否已经保存 Key。
- Base URL 是否为不含凭证和查询参数的绝对 HTTP(S) URL。

### 返回 `model not found`

模型名称必须与供应商 `/v1/models` 返回的 ID 完全一致。不要把网页展示名称当成
API 模型 ID。

### 返回 Tool Calling 或 JSON 参数错误

关闭对应能力开关并重新保存。普通 Chat 成功后，再根据模型和推理服务文档逐项启用。

### 费用显示“未配置模型价格”或“API 未返回用量”

先确认最终使用的模型 ID 与 `LLM_MODEL_PRICING` 键完全一致。兼容端点若支持
`stream_options.include_usage`，再开启对应流式 usage 开关；不支持时保持关闭，
避免请求参数导致 400。费用是 Token 单价估算，实际扣费仍以供应商账单为准。

### 本地模型能从宿主机访问，但 MindForge 访问失败

确认模型服务监听 `0.0.0.0`，并从应用容器执行 `/v1/models` 连通性检查。不要把
容器内的 `127.0.0.1` 当成服务器宿主机。

### 切换 Provider 后 RAPTOR 或 GraphRAG 请求了错误模型

将 `RAPTOR_SUMMARY_MODEL`、`GRAPH_ENTITY_EXTRACTION_MODEL` 和
`GRAPH_COMMUNITY_SUMMARY_MODEL` 留空，使其继承当前 Provider；显式覆盖时必须填写
当前端点真实存在的模型 ID。

## 七、安全要求

- `.env`、API Key、Token 和模型服务鉴权信息禁止提交 Git。
- Base URL 禁止嵌入用户名或密码。
- 设置 API 返回的 Key 是脱敏值，前端不会持久化完整 Key。
- 不要直接将无鉴权的本地推理端口暴露到公网。
- 远程部署时先备份服务器 `.env`，再按键合并新增配置。
