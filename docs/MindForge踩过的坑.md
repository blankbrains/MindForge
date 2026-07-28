# MindForge 踩过的坑

> 这些问题都是我在开发 MindForge（Multi-Agent RAG 研究助理系统）时真实遇到、排查并修复的。写在这里，既是为面试准备，也是给自己留个记录。

---

## 一、LLM 调用链路

### 1. API Key 保存了但实际没生效

**现象**：在设置页面配置了 DeepSeek API Key，后端也返回 `{"status":"saved"}`，但发起研究请求时一直报 401。

**根因**：`update_settings_api` 把 key 加密存入了数据库，但**没有写回 `os.environ`**，而 `LLMFactory.create()` 只从 `get_settings()` 读取——`get_settings()` 又是通过 `lru_cache` 缓存的。所以：

```
PUT /settings → DB 有 key ✅ → 但 os.environ 没有 ❌
→ get_settings() 读到空字符串 → LLMFactory 没 key → 401
→ 服务器重启后 os.environ 丢失 → 全靠 DB fallback
```

**修复三处**：
1. 保存 key 时同步写入 `os.environ`
2. `LLMFactory.create()` 加 `_load_api_key_from_db()` 兜底（重启后从 DB 解密读取）
3. `_sync_env_file()` 同步写入 `.env` 文件

**教训**：配置的全链路一致性——DB、环境变量、文件、缓存，哪个断链都会出问题。

### 2. 脱敏 Key 覆盖了真实 Key

**现象**：反复设置 API Key 后，Key 莫名其妙变成了 `***abcd` 这样的值，DeepSeek 一直报 401。

**根因**：前端 `loadSettings` 从后端 GET 接口读到脱敏值 `***f072`，存入 `llmApiKey`。用户再次点保存时，前端把这个 `***f072` 当作真实 Key 发回后端，后端写入 `os.environ` 和 `.env`——正确的 Key 被覆盖了。

```
GET /settings → "***abcd" → 前端存为 llmApiKey
→ 用户点击保存 → PUT { deepseek_api_key: "***abcd" }
→ 后端写入 .env: LLM_DEEPSEEK_API_KEY=***abcd → ❌
```

**修复**：
- 前端：`***` 开头的值不发送（`isMasked ? undefined : state.llmApiKey`）
- 后端：收到 `***` 开头的 key 直接跳过

**教训**：脱敏值回显是安全需求，但不能因此污染真实数据。

### 3. `reload_settings` 丢失导致配置不刷新

**现象**：通过 API 切换 LLM provider 后，调用的还是旧模型。

**根因**：`get_settings()` 被 `@lru_cache` 装饰，修改完 `os.environ` 后没有清除缓存。

**修复**：添加 `reload_settings()` 函数：`get_settings.cache_clear()`。

### 3.1 Docker 单文件挂载导致 `.env` 保存返回 500

**现象**：容器内 `/app/.env` 可写，文件和目录 UID/GID 也与运行用户一致，但 `PUT /api/v1/settings` 仍返回 500，日志出现：

```text
OSError: [Errno 16] Device or resource busy:
'/app/.tmp_xxx' -> '/app/.env'
```

**根因**：Compose 使用 `./.env:/app/.env` 单文件 bind mount。`python-dotenv.set_key()` 默认先写临时文件，再用 `os.replace()` 原子替换目标；Docker 不允许替换挂载点本身，因此权限检查全部通过仍会触发 `EBUSY`。

**修复**：

1. 在 `/app` 中创建普通暂存文件，并在暂存文件上调用 `set_key()` / `unset_key()`。
2. 更新完成后加锁，以二进制方式原地覆盖已挂载的 `.env`，执行 `flush()` 和 `fsync()`，不替换 inode。
3. 保持文件权限为 `0600`，异常时清理暂存文件，并沿用数据库事务回滚逻辑。
4. 增加回归测试，模拟目标 `.env` 禁止 `os.replace()`，确认配置仍能保存且原有键不丢失。

**教训**：文件“可写”不等于文件“可被替换”。遇到容器挂载点时，需要区分内容写入、重命名和 inode 替换三种文件系统语义。

### 3.2 PostgreSQL 回退连接串会把开发凭证带入部署

**现象**：未配置 `DATABASE_URL` 时，应用仍能尝试使用代码里的固定
PostgreSQL 连接串。部署环境一旦变量遗漏，问题会表现为连接错误，且容易把开发
账号、端口或数据库名当成生产默认值。

**根因**：数据库模块为方便本地启动提供了环境变量缺失时的连接 URL 回退，这破坏了
根目录 `.env` 作为唯一运行时配置源的约束。

**修复**：

1. 通过 `require_environment_variable("DATABASE_URL")` 在数据库模块加载时校验必填配置。
2. `.env.example` 只保留明确标记为待替换的模板值，并要求 `DATABASE_URL` 与 PostgreSQL 配置保持一致。
3. 增加回归测试，确认缺失 `DATABASE_URL` 会给出可操作的配置错误。

**教训**：基础设施连接信息不能有“方便开发”的隐式默认值。部署缺配置时应尽早失败，
而不是尝试连接一个可能错误或不安全的目标。

---

## 二、Agent 编排系统

### 4. Planner DAG 死锁 vs 误杀

**现象**：有时候任务卡住了，有时候所有子任务都被标记为 failed。

**根因**：LLM 生成的 DAG 可能包含循环依赖（t1→t2→t1）或指向不存在的 task_id。原代码在 `get_ready_tasks()` 返回空时，把所有 pending 子任务全标 failed：

```python
if not ready:
    for st in plan.subtasks:
        if st.status == "pending":
            st.status = "failed"  # 连带杀掉了可独立执行的任务
    break
```

**修复**：改为只标记"依赖无法满足"的任务：

```python
unsatisfied = [dep for dep in st.dependencies
              if dep not in completed_ids
              and any(s2.task_id == dep and s2.status == "pending" for s2 in plan.subtasks)]
if unsatisfied:
    st.status = "failed"
```

**教训**：异常处理的粒度很重要，"兜底"不能变成"乱杀"。

### 5. `as_completed` 的 KeyError

**现象**：`asyncio.as_completed` 替换 `gather` 后，Agent pipeline 崩溃，日志报错 `coroutine object _wait_for_one at 0x...`。

**根因**：`asyncio.as_completed(aws)` 返回的 Future 和传入的 aws 不是同一个对象。

```python
task_map = {asyncio.create_task(self._execute_subtask(st)): st for st in ready}
for coro in asyncio.as_completed(task_map):
    st = task_map[coro]  # ❌ coro 不是原来的 task，KeyError
```

**修复**：改回 `asyncio.gather` + `return_exceptions=True`。

**教训**：标准库的高级特性（`as_completed`）在 Python 里的行为跟直觉不一样，生产环境慎用不熟悉的 API。

### 6. Critic 精炼后分数不对

**现象**：Critic 评的分和最终报告质量对不上。

**根因**：精炼循环的结构是"评估→精炼→评估→精炼..."。如果达到 `max_refine` 后循环退出，最后一次精炼**没有配套的评估**，返回的 `final_critic` 是精炼前的分数。

```python
for refine_round in range(max_refine):
    critic_score = evaluate(draft)   # 评估
    if not should_refine: break
    draft = synthesize(...)          # 精炼
    # 循环结束 → 没有对 new_draft 做评估
```

**修复**：循环结束后补做一次最终评估。

**教训**：循环的边界条件——最后一轮是否有配套的收尾动作。

### 7. Critic 失败降级触发精炼风暴

**现象**：Critic 解析 LLM JSON 失败时，返回 `should_refine=True`，导致 Synthesizer 盲目重写，重写后 Critic 又报错，白跑两轮。

```python
except Exception as exc:
    return CriticScore(overall=5.0, should_refine=True)  # ❌
```

**修复**：评估失败时返回 `should_refine=False`，信任当前 draft。

**教训**：降级策略不能和触发条件冲突。

---

## 三、检索系统

### 8. RRF 分数尺度混合

**现象**：加了 BM25 后检索质量反而下降。

**根因**：

```python
fused[doc_id]["score"] += 0.6 * rrf + 0.4 * raw_score
```

`raw_score` 有两个来源：
- 稠密向量（Qdrant）：cosine 相似度 ∈ [0, 1]
- BM25 稀疏：bm25s 原始分数 ∈ [0, 几十)

BM25 的大数压倒向量信号，RRF 融合名存实亡。

**修复**：每个 source 内先归一化 raw_score 再混合，或纯用 RRF。

**教训**：多源分数融合时，源之间的尺度一致性比算法更重要。

### 9. BM25 查询传参格式错误

**现象**：BM25 检索时报维度异常，降级到关键词匹配。

**根因**：`bm25s.retrieve` 期望查询 token 为 `[[...]]`（二维），传的是 `[...]`（一维 list）。

```python
query_tokens = list(jieba.cut(query))           # ❌ 1D
# 正确: query_tokens = [list(jieba.cut_for_search(query))]  # ✅ 2D
```

**教训**：第三方库的 API 签名要仔细看，`bm25s` 的 `retrieve` 和 `index` 的输入形状必须一致。

### 10. Qdrant point ID 不稳定

**现象**：重复索引同一文档后，Qdrant 中出现重复数据。

**根因**：用 Python 内置 `hash()` 做持久化 ID。Python 3.3+ 默认 `PYTHONHASHSEED` 随机化，同一字符串在不同进程算出的 hash 不同。

```python
id=abs(hash(ch.chunk_id)) % (2**63)  # ❌ 跨进程不一致
```

**修复**：改用 `hashlib.md5`。

**教训**：`hash()` 不能用作持久化 ID——这是 Python 基础但容易忽略的点。

### 11. GraphRAG save/load 字段不同步

**现象**：GraphRAG save 时抛 `AttributeError`，load 时抛 `TypeError`。

**根因**：`Community` dataclass 定义了 `id`、`entities`、`summary`，但 `save()` 访问 `c.entity_ids` 和 `c.label`，`load()` 又用 `entity_ids=set(...)` 构造。dataclass 和序列化逻辑不同步。

**教训**：数据模型变更后，所有序列化/反序列化代码必须同步更新。

---

## 四、SSE 流式

### 12. `[DONE]` 和 `done` 双完成信号竞态

**现象**：偶尔出现"已完成但无报告"的白屏。

**根因**：前端有三路完成信号：

1. `done` 事件（带 result 数据）
2. `[DONE]` 数据帧（纯终止符）
3. Stream reader `done`（流自然结束）

如果 `[DONE]` 先于 `done` 到达，`onComplete` 在没有 `finalResult` 的情况下就把状态置为 `"completed"`，UI 渲染 `isCompleted && finalResult` 为空。

**修复**：`onComplete` 中加守卫：`if (useResearchStore.getState().finalResult) { ... }`

**教训**：双协议标记（事件 + 标记帧）必须有明确的优先级和幂等保护。

### 13. 流式 tool_calls 未增量聚合

**现象**：流式模式下，工具调用碎片化，每次 delta chunk 都 yield 一个独立的 tool_call 事件。

**根因**：OpenAI 的流式 API 把 `tool_calls` 分段返回：第一个 chunk 带 `id` + `function.name`，后续 chunk 只有 `function.arguments` 片段。原代码直接透传没有聚合。

**修复**：按 `tc.index` 维护 accumulator：

```python
tool_acc = {}
for tc in delta.tool_calls:
    idx = tc.index
    slot = tool_acc.setdefault(idx, {"id": None, "function": {"name": None, "arguments": ""}})
    if tc.id: slot["id"] = tc.id
    if tc.function.arguments: slot["function"]["arguments"] += tc.function.arguments
# 流结束后一次性发出完整 tool_calls
```

**教训**：流式 API 的增量语义和一次性语义是不同的，Consumer 不能假设两边格式一样。

---

## 五、前端

### 14. API Key 警告恒显

**现象**：即使配了 Key，研究页面一直在顶部显示黄色警告"未配置 LLM API Key"。

**根因**：`research-page.tsx` 中 `{(<div>...</div>)}`——外层 `()` 没有条件判断，恒为真。这是 TSX 语法错误，不是运行时 bug，但说明代码审查不严。

**修复**：改为 `{!hasLLMKey && (...)}`，由 `settings-store.hasLLMKey` 驱动。

### 15. 重试按钮传空 task

**现象**：研究出错后点"重试"，什么都没发生。

**根因**：`onSubmit` 里 `setTask("")` 把输入框清空。重试按钮 `onClick={() => session.startResearch(task)}` 读到的 `task` 是空字符串。

```tsx
onSubmit={(t) => { session.startResearch(t); setTask(""); }}  // 提交即清空
onClick={() => session.startResearch(task)}                     // 重试时空 task
```

**修复**：用 `lastTaskRef` 保留最后一次提交的任务。

### 16. DAG 节点状态不更新

**现象**：流式过程中，子任务状态变了，但 React Flow 的节点颜色不变。

**根因**：`useMemo` 的依赖只包含 `[plan]`，而 `plan` 对象引用在 `plan_ready` 后不变。子任务状态存在 `subtasks` 字典中（流式更新），但 `useMemo` 没有把它作为依赖。

**修复**：`useMemo` 的 dep 改为 `[plan, subtaskStates]`，从 `useResearchStore` 读取实时 subtask 状态。

### 17. 历史 id 用浮点数

**现象**：删除某条历史记录时后端 404。

**根因**：`Date.now() + Math.random()` 产生浮点数 id（如 `1718000000000.4231`），后端按整数解析失败。

**修复**：`Math.floor(Date.now() * 1000 + Math.random() * 1000)`，优先用后端返回的整数 id。

---

## 六、MCP 协议

> 本节记录历史 MCP 实现中的真实问题。当前 `main` Web 应用已停用 MCP，
> 不暴露 HTTP 入口、不在启动阶段加载，也不向 Researcher 注册 MCP 工具。

### 18. JSON-RPC request id 用内存地址

**现象**：并发 MCP 请求响应串扰。

**根因**：`id=str(id(params) if params else 0)`——`id()` 返回对象内存地址，进程内可能重复；无参请求时 id 全为 `"0"`。

**修复**：改用 `uuid.uuid4().hex[:8]`。

### 19. `run_research_task` 参数签名不匹配

**现象**：通过 MCP 调用研究任务永远失败。

**根因**：MCP server 调用 `self._research_agent.run(topic=topic, depth=depth, max_sources=max_sources)`，但 `ResearcherAgent.run` 的签名是 `run(task)`。

**修复**：MCP 端拼接参数为描述字符串后传给 `run(task_desc)`。

---

## 七、性能优化

### 20. 空知识库时 Agent 反复重试

**现象**：研究"什么是 agent"用了 2 分钟，最后说"暂无相关资料"。

**根因**：Researcher 每次 ReAct 都先搜知识库——空结果 → 换关键词再搜 → 空结果 → 再搜...直到用完 max_iterations。18 次 LLM 调用全部花在"搜不到"上。

**修复**：
1. Researcher prompt 加规则：1-2 轮搜索无果后直接用自身知识回答
2. Synthesizer prompt 加规则：sparse findings 时生成 500+ 字完整回答
3. 参数调优：`max_iterations=8→3`，`max_refine_rounds=2→1`

---

## 八、生产化与安全

### 21. 配置散落导致本地、Docker、Vite 行为不一致

**现象**：同一个参数在 Python 默认值、Compose、Vite 和启动脚本中各有一份，修改后只在部分运行方式生效。

**根因**：缺少单一配置源，结构文件里还混入了部署环境值。

**修复**：根目录 `.env` 作为唯一运行时配置源，`.env.example` 保持完整键集合；Pydantic Settings、Vite `envDir`、Docker Compose、启动脚本和 QA 脚本都读取同一个文件。

### 22. `/health` 只返回进程存活，依赖挂了仍显示正常

**现象**：FastAPI 能响应，但 PostgreSQL、Redis 或 Qdrant 不可用，监控仍判定服务健康。

**根因**：健康检查没有执行真实连接探测，也没有区分存活与就绪。

**修复**：`/health` 返回真实核心依赖状态；`/ready` 在任一核心依赖失败时返回 HTTP 503，Dockerfile 和 `start.sh` 都使用就绪端点。

### 23. 大文件上传和恶意 DOCX 可能耗尽内存

**现象**：上传接口一次性读取文件，压缩包格式还可能通过高压缩比制造解压放大。

**根因**：只校验扩展名，没有对流式大小、PDF 页数、DOCX 解压体积、部件数和解析后字符数建立边界。

**修复**：上传内容流式写入临时文件，并由 `API_*` 配置限制上传体积、文本体积、PDF 页数、DOCX 解压体积/部件数、解析字符数和文档 Chunk 数。

### 24. 代码执行器在主进程中运行，超时无法真正终止

**现象**：用户代码死循环、打印海量输出或尝试文件/网络操作时，会拖住 API 进程。

**根因**：线程超时只能停止等待，不能可靠终止执行中的 Python 代码。

**修复**：改为隔离子进程，设置 CPU、地址空间、代码长度、变量体积和输出上限，并通过导入白名单和审计 Hook 拒绝文件、进程、网络等危险操作。

### 25. GraphRAG 删除文档后共享实体来源失真

**现象**：两个文档共享实体时，删除其中一个会误删实体，或者保留已经不存在的来源关系。

**根因**：实体只保存聚合状态，没有在删除时重新计算文档来源和关系权重。

**修复**：删除文档后重建共享实体 provenance，保守处理旧版无来源字段的数据，并对关系权重执行有限值和边界校验。

### 26. 追踪日志可能记录 API Key 或完整用户内容

**现象**：调试追踪虽然方便，但异常参数、Authorization、Token 或 Prompt 可能落入 JSONL/LangFuse。

**根因**：缺少统一的敏感字段识别、值脱敏和内容采集开关。

**修复**：Tracer 对 key/password/secret/token/cookie 等字段和疑似凭证值统一脱敏；默认 `OBSERVABILITY_CAPTURE_CONTENT=false`，并限制单条记录、单文件大小和保留天数。

### 27. 本地 `.env` 覆盖服务器配置导致容器无法启动

**现象**：代码同步后重新构建，PostgreSQL 因主机端口被占用而启动失败；修复端口后，应用又只绑定到 `127.0.0.1`。容器内进程还可能因 UID/GID 不匹配而无法读取绑定挂载的 `.env`。

**根因**：本地与服务器虽然共享同一套配置键，但部署值不同。把本地 `.env`
整文件上传后，覆盖了服务器专用的 PostgreSQL 端口、应用绑定地址以及
`APP_UID/APP_GID`。宿主 `.env` 属于原用户，容器 UID 改变后即使文件保持
`600`，运行时配置接口也无法读写。

**修复**：远程部署只同步代码和文档；`.env` 先备份再按键合并。配置变更
只更新相关键，并在重建前运行 `docker compose config --quiet`。容器
UID/GID 必须与 `.env` 文件所有者对齐；调整时先迁移数据卷和模型缓存所有权。
重建后同时检查 Compose 端口映射、`/api/v1/ready` 和远程访问地址。

### 28. 旧 SPA 缓存叠加设置页懒加载导致卡死

**现象**：点击侧边栏“设置”后长期停在“正在加载页面”，部分 Edge 渲染进程
随后以 `STATUS_ILLEGAL_INSTRUCTION` 崩溃。移除 MCP 管理后，已经打开的旧
浏览器标签页仍可复现。

**定位证据**：设置 API 始终快速返回 200，响应不足 1 KB，服务端无异常。干净
浏览器会请求当前 `settings-page-*.js` 并正常进入设置页；发生卡死的浏览器仍
持续轮询 `/health`，但点击设置时服务端没有收到当前设置 chunk 请求，说明它
继续运行部署前缓存的 SPA 和旧 chunk。

**根因**包含两个阶段。第一阶段是设置页使用 TanStack Router 的
`lazyRouteComponent`，旧标签页继续运行部署前入口；同时
`importWithReload` 在导入失败后返回永不结束的 Promise，导致路由永久停在
pending。取消动态 chunk 后，用户 Edge 仍出现原生崩溃，Crashpad 转储进一步
定位到 `dwrite.dll`，异常码 `0xc000001d`，多次崩溃地址完全一致。

最终触发条件是设置 Store 持久化了 `hasLLMKey=true`，但不会持久化真实或脱敏
`llmApiKey`。设置页在 API 返回前立即渲染表单，于是旧浏览器状态会先在
`font-mono` 元素中显示中文占位文本，触发该机器 Edge/DirectWrite 的字体回退
崩溃。干净浏览器没有这份持久化状态，因此一直无法复现。

**修复**：设置页改为同步路由，构建产物不再生成 `settings-page-*.js`；桌面和
移动端设置链接使用 `reloadDocument` 强制完整文档导航；动态导入失败后触发
重载并抛出原错误，不再留下永久 pending Promise。HTML 保持 `no-store`，哈希
资源保持 `immutable`。设置数据加载完成前只显示加载态，失败时显示可重试
错误态；API Key 展示和输入取消等宽字体，空值仅使用 ASCII 掩码。设置 Store
迁移到 `mindforge-settings-v2`，忽略旧持久化状态，并由 AppShell 启动时从
服务器统一初始化。同时移除主页 MCP 状态卡和管理界面。后续进一步移除了
`/api/v1/mcp`、启动加载、健康字段、Agent 工具注册、CLI 入口和 `.env.example`
配置；遗留协议模块仅用于历史学习。

### 29. `AgentResult(success=False)` 被包装成成功响应

**现象**：非流式 `/query` 中，Orchestrator 明确返回失败结果，API 仍返回 200
和普通 `QueryResponse`，调用方无法区分成功报告与失败文本。

**根因**：路由只捕获异常，没有检查 `result.success`。

**修复**：失败结果进入纯检索 fallback；fallback 也失败时返回 503。回归测试用
假的 Orchestrator 固定复现，不依赖真实 LLM。

### 30. 情节记忆用关键词重叠跳过新研究

**现象**：缓存过 `explain react hooks` 后，请求
`explain react hooks security` 会直接返回旧答案。

**根因**：`recall()` 允许两个词重叠就视为命中，而且内存副本没有执行 TTL。

**修复**：自动复用只允许规范化后的任务完全相同，并在查询前清理过期记录。
模糊检索仍可用于分析或推荐，但不能替代新任务执行。

### 31. 引用验证只看编号，不看来源是否支持声明

**现象**：`The moon is made of cheese [1]` 配一条 Python 文档来源仍被判定
100% 有效。

**根因**：旧实现只检查 `[N]` 是否落在来源列表范围内。

**修复**：增加保守的声明句提取和词汇支持检查；来源只有 URL、内容为空或与
声明没有最低限度交集时标记问题。该工具仍不是事实核查器，文档中不再宣称
未经基准验证的幻觉率。

### 32. 索引链路会静默接受空文档和向量截断

**现象**：空文档返回 `indexed`；Embedding 返回数量少于 Chunk 时，
`zip(chunks, vectors)` 静默丢掉尾部数据。

**根因**：两个索引入口各写一套流程，缺少统一的空值、数量和维度校验。

**修复**：`/index` 与 `/upload` 共用索引管线；空文档返回 422，向量数量或
维度不匹配直接失败并回滚。解析、Embedding 和同步 Qdrant 初始化移到
`asyncio.to_thread()`，避免阻塞单 worker 事件循环。

### 33. RAPTOR 重复计算叶子 Embedding

**现象**：上传流程先让 RAPTOR 对每个叶子逐条 `embed_single()`，随后又对所有
Chunk 批量 Embedding。

**根因**：增强索引执行顺序早于基础向量生成。

**修复**：先批量生成并写回 `DocumentChunk.embedding`，RAPTOR 直接复用叶子
向量，仅为摘要节点生成新向量；聚类和同步模型调用移出事件循环。

### 34. LLM 切换顺带切换 Embedding，污染已有 Collection

**现象**：设置页切到 OpenAI/DeepSeek 时自动改 Embedding provider，旧向量不
重建；显式后端初始化失败还会静默退化为 hash 向量。

**根因**：把 LLM 和 Embedding 当成同一个供应商开关，忽略向量空间兼容性。

**修复**：前端不再随 LLM 自动发送 `embedding_provider`；已有索引时后端拒绝
切换 Embedding。显式 provider 不可用时停止索引，不写 fallback 向量。

### 35. 文档内容预览重复 overlap 且 payload 被截断

**现象**：内容预览把重叠 Chunk 用空行拼接，重复显示文本；每个 payload 又只
保存前 2000 字，合法的 2048 字 Chunk 会丢内容。

**根因**：索引写入与文档重建没有使用 `chunk_start/chunk_end`。

**修复**：保存完整 Chunk 和结构化 metadata；预览按字符区间去重拼接。历史
时间同时统一补 UTC `Z`，避免浏览器把无时区时间误当成本地时间。

### 36. SPA fallback 把不存在的 API 返回成 HTML 200

**现象**：访问 `/api/v1/not-a-real-endpoint` 得到前端 `index.html` 和 200。

**根因**：通配 SPA 路由没有排除 `/api/`。

**修复**：未知 API 路径返回 JSON 404，只有非 API 路由才回退到 SPA。

### 37. 旧 SSE 连接回调污染新研究会话

**现象**：快速开始第二个任务后，第一个连接迟到的 chunk、完成或错误事件仍会
写入当前 Zustand 状态；前端超时还固定为构建时常量。

**根因**：客户端只有一个 AbortController 引用，没有请求代次标识。

**修复**：每次开始/取消研究递增 generation id，所有回调先检查代次；研究总
超时优先读取运行时设置。Vitest 直接模拟两个连接，验证旧回调被忽略。

### 38. 设置更新缺少统一锁和资源释放

**现象**：并发保存可能让 DB、`.env`、`os.environ` 和单例处于不同版本；配置
切换后旧 Qdrant、Redis、OpenAI 客户端保持打开。

**根因**：只有 `.env` 文件锁，没有覆盖完整设置事务。

**修复**：设置更新使用进程级重入锁串行执行，保留 `.env`/DB 回滚；提交成功后
重载配置并关闭旧 Orchestrator、Embedding 和 Qdrant 资源。

### 39. 上传取消弹窗被覆盖，构建上下文携带陈旧 egg-info

**现象**：上传中点击关闭后，确认弹窗和上传弹窗同为 `z-50`，后渲染的上传框
盖住确认框；Docker 构建还会复制本地 `src/mindforge.egg-info`。

**根因**：Modal 渲染顺序错误，`.dockerignore` 未覆盖 `*.egg-info`。

**修复**：取消确认放到上传弹窗之后渲染，取消异常触发服务端尽力回滚；
`.dockerignore` 增加 `*.egg-info`。

### 40. BM25 声明可用但生产依赖缺少 jieba

**现象**：`bm25s` 可导入，中文查询却一直退化为关键词匹配。

**根因**：中文分词依赖只写在代码分支，没有进入生产依赖锁。

**修复**：将 `jieba` 加入 `pyproject.toml` 和哈希锁，并增加依赖可用性回归。

### 41. Chunk 去空白后仍保留原始偏移，预览吞空格

**现象**：`alpha beta gamma` 重建后变成 `alpha betagamma`。

**根因**：Chunk 内容被 `strip()`，`chunk_start/chunk_end` 却仍指向原文。

**修复**：固定分块保留原始切片；语义分块按原文 span 记录准确偏移，并做
round-trip 测试。

### 42. 流式失败结果和缺 Key 初始化没有进入检索回退

**现象**：非流式可回退，SSE 收到 `success=False` 或 Orchestrator 因缺 Key
初始化失败时却直接报错。

**根因**：初始化发生在 fallback 的 `try` 外，SSE 只捕获异常、不检查失败
`done` 结果。

**修复**：初始化纳入回退边界；SSE 识别失败 `done` 并统一走异步 RAG fallback。

### 43. 同步检索和 GraphRAG 扫描阻塞事件循环

**现象**：BM25、CrossEncoder、RAG 初始化和 GraphRAG 分词/全图扫描均运行在
事件循环线程。

**修复**：同步工作统一移入 `asyncio.to_thread()`；GraphRAG 用操作锁串行化
构建、查询和删除，并用状态锁保护同步查询。

### 44. BM25 查询和重建并发时结果映射到错误文档

**根因**：查询使用旧索引返回位置后，读取的却是已经被重建替换的新文档数组。

**修复**：查询、构建、保存和加载使用同一实例重入锁，确保索引与文档快照一致。

### 45. 设置页跨供应商草稿丢失，删除 Key 被无关非法值阻断

**修复**：保存时提交两个供应商的全部草稿；删除 Key 只发送目标供应商空值，
不携带其他设置。

### 46. CodeExecutor 安全加固后科学库全部不可用

**现象**：`numpy/pandas/scipy/sklearn` 在白名单中，但导入时报
`ctypes.dlopen` 或 Windows `kernel32` 错误。

**根因**：审计 Hook 在可信科学库加载本地扩展前就禁止动态库初始化。

**修复**：只预加载用户代码静态声明的白名单模块，再安装审计 Hook；同时禁止
用户访问 `ctypeslib/CDLL/windll` 等桥接属性。子进程环境只继承必要系统变量，
不继承 API Key。

### 47. CodeExecutor 中文异常触发父进程解码崩溃

**修复**：子进程强制 UTF-8，父进程使用替换解码，并把空 stdout 和外层异常
转换为失败 `ToolResult`。

### 48. RAPTOR 相同摘要跨文档覆盖

**根因**：摘要节点 ID 只依赖层级、聚类序号和摘要文本。

**修复**：ID 加入所有 child node ID 的稳定指纹，跨文档节点集合不再相交。

### 49. 文档和历史详情的迟到响应覆盖当前选择

**修复**：文档预览使用 AbortController 和请求身份校验；历史详情使用 generation
id，关闭、删除和清空时使旧请求失效。

### 50. Agent 空文本和 Synthesizer 流协议被误判成功

**现象**：LLM 返回空文本时 Agent/Synthesizer 仍 `success=True`；多子任务流式
综合直接对未 await 的协程执行 `async for`。

**修复**：空白输出统一标记失败，Orchestrator 初次综合失败时停止，精炼失败时
保留上一版草稿；流式 chat 先 await 得到异步迭代器。

### 51. WebSearch 在干净生产环境中两条后端都不可用

**根因**：缺少 `tavily-python` 和 `requests` 生产依赖；注入客户端还被 SDK
缺失判断提前忽略，DuckDuckGo 正则无法解析重定向链接。

**修复**：锁定正式依赖，增加运行时参数校验和 Tavily 异常回退，并用
BeautifulSoup 解析 DuckDuckGo。

### 52. 大 PDF 跨线程共享同一解析对象，错误被吞成空页

**修复**：先改为每个 worker 独立打开 PDF 并处理连续页范围，再用真实 523 页
文件比较线程与 `spawn` 进程。8 线程为 `<thread-baseline>`，12 进程为 `<process-baseline>`，
最终默认多进程并保留线程回退；单页失败记录具体页码。

### 53. 设置重置误删 DeepSeek Key，资源关闭失败阻断新配置

**根因**：重置按钮先切到 DeepSeek 再清 Key；运行时重载把所有 reset 串在一次
无保护调用中。

**修复**：重置只恢复非敏感参数，所有 Key 草稿保持不变；旧资源关闭和各单例
重建逐项隔离，某一项失败不会阻断其余配置生效。

### 54. 移动端首次加载自动展开侧边栏遮挡主内容

**现象**：在 390px 移动视口全新打开首页时，侧边栏和遮罩立即覆盖主内容，
用户必须先关闭抽屉才能使用页面。

**根因**：全局 UI Store 将 `sidebarOpen` 无条件初始化为 `true`；桌面布局虽然
不读取该状态，但移动布局会据此直接渲染抽屉。

**修复**：将抽屉初始状态改为关闭，保留菜单按钮显式打开行为，并增加 Store
回归测试和真实移动视口浏览器验证。

### 55. 非流式检索降级把 Qdrant 异步客户端带到错误事件循环

**现象**：缺少 LLM Key 时 `/query` 表面返回 200，但日志出现
`bound to a different event loop`，向量检索实际失败后被混合检索降级逻辑吞掉。

**根因**：API 将 `RAGTool.safe_execute` 放入工作线程；同步桥接在该线程中创建
新的事件循环，却复用了主事件循环初始化的 Qdrant 异步客户端。

**修复**：非流式 fallback 直接 `await RAGTool.execute_async()`，仅由工具内部
卸载同步初始化，异步 Qdrant 调用始终留在请求事件循环。现有 fallback 回归测试
同时禁止再次调用同步桥接。

### 56. 缺少 LLM Key 的正常降级被记录为 ERROR 堆栈

**现象**：系统按设计在未配置 API Key 时返回知识库检索结果，但每次请求都会
输出 ERROR 和完整 Traceback，导致监控误报并掩盖真实故障。

**根因**：模型适配器用通用 `ValueError` 表示缺少 Key，API 又对所有初始化异常
统一调用 `logger.exception`。

**修复**：新增 `LLMConfigurationError`，OpenAI/DeepSeek 缺 Key 时统一抛出；
API 对该明确配置状态记录 WARNING，未知异常仍保留 ERROR 堆栈，回归测试同时
校验降级成功和日志级别。

### 57. 大 PDF 超过页数上限却显示“服务器繁忙”

**现象**：上传约 300 页的书籍 PDF 后，前端只提示“服务器繁忙，请稍后重试”。
实测文件实际有 523 页，而服务端原上限为 500 页。

**根因**：解析器用通用 `ValueError` 表示页数超限，API 没有把输入边界异常
映射到 4xx，前端又把所有 5xx 统一遮蔽成服务器繁忙。

**修复**：增加文档解析异常层级，将页数、字符数和 Chunk 数等限制映射到
400/413/422；错误信息同时包含实际值和配置上限。前端优先展示服务端 `detail`，
知识库上传框显示当前 PDF 页数上限。大体量 PDF 测试 在上限 600 时
成功生成 925 个 Chunk，601 页测试文件返回 HTTP 413。

### 58. 异步任务临时文件名导致相同 PDF 重复索引

**现象**：同一份 大体量 PDF 重复上传后，文档数从 1 增长到 2，Chunk 从
925 增长到 1850，并再次消耗约 223 秒执行 CPU Embedding。

**根因**：旧 `doc_id` 包含文件名、mtime 和内容前缀。异步任务每次都给上传
文件加新的 job ID 前缀，所以相同内容必然得到不同文档 ID；BM25 也只能按新的
Chunk ID 追加。

**修复**：文档 ID 改为解析内容 SHA-256；新增索引签名覆盖分块、Embedding、
RAPTOR 和 GraphRAG 配置。命中前核对 PostgreSQL、Qdrant 和 BM25 Chunk 数；
Qdrant 重建时替换同文档 Point，BM25 按文档整体替换。服务器重复上传实测
`<reuse-duration>`，未进入 Embedding，文档和 Chunk 总数不再增长。

### 59. 服务器有 GPU，但容器只能使用 CPU

**现象**：宿主机存在 NVIDIA GPU，但生产容器运行 `torch==2.13.0+cpu`，
大体量 PDF 的 BGE-M3 Embedding 仍需 `<cpu-baseline>`。服务器没有
NVIDIA Container Toolkit，常规 `gpus: all` 配置无法工作。

**根因**：GPU 是否存在和容器是否获得 CUDA 设备是两件事。CPU Torch wheel
不包含 CUDA 后端，Docker 默认也不会把 `/dev/nvidia*` 和宿主机驱动库暴露给
容器。

**修复**：保留 CPU/GPU 两套互斥哈希锁；GPU Compose override 显式映射
`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`、
`/dev/nvidia-uvm-tools`、`libcuda.so` 和 `libnvidia-ml.so`，并把
Embedding 设备设为 `cuda`、batch size 设为 32。容器内真实 CUDA 矩阵运算
通过，大体量 PDF 默认 `auto` 策略完整索引降至 `<gpu-baseline>`。

该方式依赖宿主机驱动库的真实版本路径，驱动升级后需要更新服务器 `.env`。
如果目标主机具备 NVIDIA Container Toolkit，应优先使用标准 GPU runtime，
减少对宿主机文件布局的耦合。

---

## 总结

| 类别 | 坑数 | 最深的坑 |
|------|:----:|---------|
| LLM 调用链路 | 4 | 脱敏 Key 覆盖真实 Key |
| Agent 编排 | 6 | 空响应和流式综合协议错误 |
| 检索与索引 | 14 | 向量空间污染、事件循环错用、竞态与静默截断 |
| SSE 流式 | 5 | 失败结果未触发检索回退 |
| 前端 | 11 | 旧响应覆盖、Key 草稿、重置误删除和移动抽屉遮挡 |
| MCP 协议 | 2 | JSON-RPC id 内存地址 |
| 性能 | 5 | 大 PDF 解析、重复 Embedding、GPU 容器直通与同步重任务阻塞事件循环 |
| 生产化与安全 | 12 | 沙箱边界、依赖缺失、输入边界、日志误报与资源释放 |

总共 **59 个坑**，每个都落地到代码或测试层面解决了。面试官问到的时候，
挑 2-3 个讲清楚，比一口气列完更有说服力。
