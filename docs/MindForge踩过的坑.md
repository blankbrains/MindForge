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

---

## 总结

| 类别 | 坑数 | 最深的坑 |
|------|:----:|---------|
| LLM 调用链路 | 3 | 脱敏 Key 覆盖真实 Key |
| Agent 编排 | 4 | `as_completed` Future 映射失败 |
| 检索系统 | 4 | RRF 分数尺度不匹配 |
| SSE 流式 | 2 | `[DONE]` 和 `done` 竞态 |
| 前端 | 4 | 重试按钮空 task |
| MCP 协议 | 2 | JSON-RPC id 内存地址 |
| 性能 | 1 | 空知识库重试风暴 |
| 生产化与安全 | 6 | 代码沙箱隔离与输入资源边界 |

总共 **26 个坑**，每个都落地到代码层面解决了。面试官问到的时候，挑 2-3 个讲清楚，比一口气列完更有说服力。
