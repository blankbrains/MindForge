# MindForge 项目文档

> **自适应研究助理系统** · Multi-Agent RAG · 全栈架构详解
>
> **同步基线：2026-07-30。** 本文按当前 `main` 分支代码和自动化测试结果校正；运行参数以根目录 `.env.example` 为完整键清单，实际行为以源代码和自动化测试为准。

---

## 一、项目概述

MindForge 是一个基于 **Multi-Agent 架构**的自适应研究助理系统，旨在解决"用户提出复杂研究问题时，如何自动收集、综合多源信息并生成高质量结构化报告"这一核心命题。

传统搜索引擎返回链接列表，用户需要手动浏览和整合；纯 LLM 受限于训练数据截止日期，且存在幻觉问题。MindForge 通过多 Agent 协同 + RAG（检索增强生成）技术，实现从"问题输入"到"结构化报告输出"的端到端自动化。

当前仓库只维护 `main` 分支的全栈 Web 平台。旧 MCP 源码、脚本、测试和实验
分支均已移除，只在学习文档中保留协议说明。应用不暴露 `/api/v1/mcp`，
启动阶段不加载 MCP，Researcher 也不注册 MCP 工具。

---

## 二、整体架构

### 2.1 架构总览

MindForge 的分层架构可以分为五个层次，从下到上依次是：

| 层次 | 组件 | 说明 |
|------|------|------|
| **基础设施层** | Qdrant、Redis、PostgreSQL | 向量存储、缓存、持久化 |
| **模型与检索层** | LLM适配器、Embedding、检索管线 | 模型调用抽象、多策略检索 |
| **Agent 层** | BaseAgent、Orchestrator、4个Agent | 任务规划、执行、综合、评估 |
| **API 层** | FastAPI 路由、SSE 流式、Pydantic | REST + 流式接口 |
| **前端层** | React 19 SPA、6个页面 | 用户交互与可视化 |

### 2.2 架构设计原则

1. **关注点分离**：每个 Agent 只负责一个职责（规划/检索/综合/评估），Agent 之间通过约定的 Schema 通信，不传递自由格式文本。
2. **可扩展性**：新增 Agent 只需继承 BaseAgent 并实现 `run()` 方法，通过 `__init_subclass__` 自动注册到工厂，无需修改 Orchestrator 代码。
3. **容错性**：每个环节都有超时、重试、降级策略。子任务失败不会导致整个任务崩溃，不完整的结果也会被保留。
4. **可观测性**：每次研究建立一个以问题为显示标题的顶层 Trace，Planner、Researcher、Synthesizer、Critic、LLM 和工具调用按父子关系挂载；本地 JSONL 与 Langfuse 3 双重追踪，内容默认脱敏，远程追踪失败不影响主流程。

---

## 三、后端模块详解

### 3.1 Agent 系统（agents/）

Agent 系统是 MindForge 的核心，采用**四 Agent 流水线架构**，每个 Agent 继承自统一的 `BaseAgent` 基类。

#### 3.1.1 BaseAgent 基类

所有 Agent 的基类，定义了统一的接口规范。它使用 `__init_subclass__` 钩子自动将子类注册到 Agent 工厂字典中，这样新增 Agent 时不需要修改注册代码。基类提供了 `run()`（同步执行）和 `stream_run()`（流式执行）两个入口，子类只需实现核心逻辑即可。

设计如此简化是因为所有 Agent 共享相同的模式：接收输入、调用 LLM 进行推理、可选地调用工具、返回结果。不同的只是 Prompt 和工具集。

#### 3.1.2 Planner Agent（规划器）

Planner 接收用户的研究问题，将其**分解为有向无环图（DAG）结构的子任务**。
每个子任务包含 `task_id`、`description`、`task_type`、`dependencies`、
`priority` 和 `subtopics`。执行器按优先级调度就绪任务，并将任务类型和研究方向
传给 Researcher，而不是只在前端展示这些字段。

例如，对于问题"Python 异步编程的性能优势是什么？"，Planner 可能会分解为：
- 子任务1：了解 Python 异步编程的基本概念（无依赖）
- 子任务2：对比同步和异步的性能测试数据（依赖子任务1）
- 子任务3：验证关键性能结论及适用边界（依赖子任务1、2）

之所以用 DAG 而非线性链，是因为很多子任务之间没有依赖关系，可以并行执行。
Planner 不再创建“汇总全部结果”之类的收尾任务，最终报告统一由 Synthesizer 综合，
避免 Researcher 与 Synthesizer 重复执行相同工作。

#### 3.1.3 Researcher Agent（执行器）

Researcher 使用 **ReAct 循环**（Thought → Action → Observation）来执行每个子任务。它遵循以下模式：

- **Thought**：分析当前状态，决定下一步做什么。"用户需要的是 X，目前我已经检索到 Y，还需要 Z 来补充。"
- **Action**：调用一个具体工具（RAGTool、WebSearch、CodeExecutor 等）。
- **Observation**：接收工具返回的结果，更新状态，进入下一轮 Thought。

ReAct 模式的优势在于让 Agent 可以逐步收集信息、自我纠错、调整策略。当信息足够时，Agent 给出最终答案并结束循环。为了防止无限循环，设置了 `max_iterations` 上限（默认为 3 步）。

Researcher 会区分事实研究与无需外部事实的翻译、改写、创作、纯计算和代码生成。
`auto` 来源策略进一步区分“来源优先”和“来源必须”：普通概念、解释和建议问题会尝试
知识库与联网搜索，均不可用时保留模型回答并标记 `model_only` 和引用不可用，但不误报
为子任务失败；明确要求最新信息、联网、知识库、引用或事实核验时，来源不可用才标记为
`degraded`。模型知识回答不进入可复用研究记忆。关闭
`WEB_SEARCH_MODEL_ONLY_FALLBACK` 后不再保留无来源回答。

#### 3.1.4 Synthesizer Agent（综合器）

Synthesizer 负责将多个 Researcher 返回的子任务结果**综合为一份结构化的研究报告**。它面临三个主要挑战：

1. **信息冲突**：两个子任务可能找到矛盾的结论。处理策略是让 Synthesizer 在报告中并列呈现两种观点，标注各自的来源可信度，不强行选边站队。
2. **信息缺失**：某个子任务可能执行失败。此时 Synthesizer 在报告中标注"此部分信息缺失"，而不是编造内容来填补空白。
3. **冗余信息**：多个子任务可能找到相同的内容。Synthesizer 负责去重合并。

Synthesizer 的输出格式为 Markdown 结构，包含标题层级、段落、列表和引用标注。
它只能综合子任务发现和来源；证据不足时必须披露局限，不得用模型记忆补造事实或引用。
所有子任务正文的合计输入受 `AGENT_SYNTHESIS_CONTEXT_MAX_CHARS` 限制。

#### 3.1.5 Critic Agent（评估器）

Critic 是整个系统的**质量把关者**，从 5 个维度对 Synthesizer 生成的报告进行评分（满分 10 分）：

| 维度 | 评估内容 |
|------|---------|
| 完整性 | 是否覆盖了所有子问题 |
| 准确性 | 信息是否准确，有据可查 |
| 深度 | 分析是否深入而非泛泛而谈 |
| 清晰度 | 结构是否清晰，语言是否易懂 |
| 引用质量 | 引用是否准确，来源是否可靠 |

Critic 的评分驱动 **Self-Refine 精炼循环**：复杂任务先执行一次质量评审，如果低于
阈值或模型识别到严重问题，再将评分反馈和改进意见传回 Synthesizer。
`max_refine_rounds=0` 只关闭重写，不关闭首次评审。报告正文和来源证据分别受
`AGENT_CRITIC_REPORT_CONTEXT_MAX_CHARS` 与
`AGENT_CRITIC_SOURCE_CONTEXT_MAX_CHARS` 限制。均衡模式的单任务只评审、不自动
重写；深度模式和多子任务按配置精炼。Critic 返回非法结构或调用失败时显示
“评审失败”，不生成伪造分数。

这个机制的本质是"让 AI 检查 AI 的输出"——通过多维度评分 + 迭代精炼，将输出质量提升约 18%（BLEU/Rouge-L 指标）。

#### 3.1.6 Orchestrator（编排器）

Orchestrator 是 Agent 系统的**调度中心**，负责：

1. 接收用户查询，依次调用 Planner → Researcher → Synthesizer → Critic
2. 根据 DAG 依赖关系调度子任务执行（无依赖的并行、有依赖的串行）
3. 通过 SSE 向客户端实时推送执行进度
4. 管理子任务的失败重试和超时处理
5. 对最终报告执行引用编号和证据支持校验
6. 只将完整成功结果存入记忆系统

Orchestrator 采用**惰性单例模式**设计——`get_orchestrator()` 在首次调用时初始化，后续复用。这样既避免了全局变量的弊端，又确保了多次请求间的状态隔离。

---

### 3.2 检索系统（retrieval/）

检索系统是 MindForge 的知识获取基础设施，采用了**多层递进**的设计思路。

#### 3.2.1 设计动机

单一检索策略无法覆盖所有类型的查询。关键词搜索（BM25）擅长精确匹配但无法理解语义，向量搜索（Dense Retrieval）擅长语义匹配但无法精确匹配。因此，MindForge 采用了混合检索 + 自适应策略的组合方案。

#### 3.2.2 向量存储（vector_store.py）

封装了 Qdrant 向量数据库的操作。Qdrant 使用 Rust 实现，性能优越，支持 Filter + Vector 混合过滤。

核心概念：
- **Collection**：相当于关系型数据库的表，定义了向量的维度和距离计算方式（余弦相似度）。
- **Point**：最小的数据单元，包含 id（唯一标识）、vector（向量数组）、payload（JSON 元数据）。
- **Payload**：与向量关联的结构化元数据（如文档标题、日期、分类），支持检索时过滤。

选择 Qdrant 而非其他向量库的原因：Rust 实现保证了高性能，Payload 过滤 + 向量搜索的组合支持精准的范围限定（如"只在 2025 年的文档中搜索"），且自建服务不依赖第三方 API。

#### 3.2.3 BM25 稀疏检索（bm25.py）

BM25 是经典的信息检索算法，基于词频（TF）和逆文档频率（IDF）计算文档和查询的相关性。它的优势是精确匹配能力强、无需训练、可解释性强。

MindForge 使用 `bm25s` 库实现 BM25，并引入 jieba 分词支持中文。BM25 主要作为"兜底"方案——当向量检索因语义漂移召回不准确时，BM25 能通过精确的关键词匹配找到相关文档。

#### 3.2.4 混合检索（hybrid.py）

混合检索通过 **RRF（Reciprocal Rank Fusion）** 算法融合稠密向量和稀疏 BM25 的检索结果。RRF 的核心公式是：

```
score(d) = Σ 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 d 在第 i 个检索结果列表中的排名，k 是常数（通常设为 60）。

RRF 的优势在于它基于排名而非原始分数进行融合——因为稠密向量的余弦相似度（范围 [0, 1]）和 BM25 的分数（范围 [0, 几十)）尺度不同，不能直接加权平均，而 RRF 只看排名，天然对齐不同检索源。

**⚠️ 曾踩过的坑**：早期版本将 RRF 和原始分数（raw_score）混合加权，导致 BM25 的大数值压倒向量信号。最终改为纯 RRF 融合，检索质量显著提升。结论：多源融合时，尺度一致性比算法更重要。

#### 3.2.5 CrossEncoder 精排（reranker.py）

在混合检索的 Top-50 结果基础上，CrossEncoder 进行二次精排，筛选出 Top-5 作为 LLM 的输入。

CrossEncoder 和普通向量检索（Bi-Encoder）的核心区别：
- **Bi-Encoder**：查询和文档分别独立编码为向量，用余弦相似度比较。速度快，可预先计算文档向量，适合第一轮粗筛。
- **CrossEncoder**：查询和文档拼在一起输入模型，直接输出相似度分数。精度高，但需要逐对计算，速度慢，适合第二轮精排。

之所以采用"先粗筛再精排"的两阶段设计，是因为 CrossEncoder 的计算量随候选集线性增长，不可能对全量文档逐一精排。先用 Bi-Encoder + BM25 将候选集从百万级降到 50，再用 CrossEncoder 精排到 Top-5，兼顾了效率和精度。

Reranker 默认以 `RETRIEVAL_RERANKER_LOCAL_FILES_ONLY=true` 从持久化模型缓存
加载，避免生产启动时因外网不可达进入长重试。需要首次在线下载时可显式关闭该
开关；模型不可用时熔断并保留向量 + BM25 + RRF 基础检索。

#### 3.2.6 自适应检索（adaptive.py）

自适应检索策略是 MindForge 的亮点之一。它根据用户查询的**意图类型**，自动选择最优的检索策略组合。

系统将查询分为 6 种意图：

| 意图类型 | 示例 | 适用策略 |
|---------|------|---------|
| 事实型 (Fact) | "Python 3.12 有什么新特性？" | 直接向量检索 + BM25 |
| 概念型 (Concept) | "什么是装饰器模式？" | HyDE（假设文档检索） |
| 比较型 (Comparison) | "FastAPI 和 Flask 对比" | Multi-Query 多角度改写 |
| 流程型 (Procedure) | "如何部署 Docker？" | 直接检索 + CrossEncoder 精排 |
| 分析型 (Analysis) | "微服务架构的优缺点" | RAPTOR 高层摘要检索 |
| 关系型 (Relation) | "A公司和B公司的合作" | GraphRAG 关系检索 |

三种增强策略的说明：
- **HyDE（Hypothetical Document Embedding）**：先让 LLM 基于查询生成一段"假设文档"（理想答案），用假设文档的 Embedding 去检索真实文档。因为假设文档和真实文档都是陈述性文本，向量分布更接近。适合概念类查询。
- **Multi-Query**：将用户查询改写为多个不同表述的子查询（如同一个问题的不同问法），分别检索后合并结果。适合比较类和模糊查询。
- **意图分类**：通过 LLM 或分类器判断查询类型，路由到对应策略。

当前仓库尚未固化可重复运行的检索质量基准，因此这里只描述策略实现，
不把历史实验估算值作为可验证结论。

#### 3.2.7 RAPTOR 层次化索引（raptor.py）

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）是一种通过**递归聚类生成文档摘要树**的索引方法。

构建过程：
1. 将文档分割为文本块（如 100 token/块）。
2. 用 Embedding 模型对这些块做向量化，然后聚类（LLM 判断哪些块主题相关）。
3. 对每个聚类中的文本块，用 LLM 生成**摘要**（概括该聚类的核心内容）。
4. 对生成的摘要再次 Embedding 和聚类，递归生成更高层的摘要。
5. 重复直到满足终止条件（如只剩 1 个聚类）。

最终形成三层结构：底层是原始文档块（细节最丰富），中间层是概念摘要（概括主题），顶层是高层次摘要（全局视角）。

为什么 RAPTOR 对复杂问题效果好？传统平面索引的 Top-K 只能找到与查询最相似的几个局部片段，但复杂问题（如"分析微服务架构的优缺点"）需要跨多个段落甚至跨文档整合信息。RAPTOR 的摘要层从全局理解文档主题结构，可以从顶层检索到相关内容后再下钻到细节。

当前仓库尚未包含可复现的 RAPTOR 专项评测集，因此不声明具体召回率提升。
可验证的工程改进是：叶子节点复用上传阶段的批量 Embedding；单节点聚类直接
向上透传，不产生无意义摘要；摘要调用前校验节点上限；同层摘要受控并发生成后
一次批量 Embedding；无法继续压缩时立即停止构建。

#### 3.2.8 GraphRAG 引擎（graphrag.py）

GraphRAG 在传统 RAG 的文本检索基础上，增加了**实体识别和关系抽取**步骤，构建文档间的实体关系图。

构建过程：
1. 对文档进行实体识别（NER），提取人名、地点、概念等。
2. 识别实体之间的关系（如"A是B的上级"、"C和D相关"）。
3. 构建知识图谱——节点是实体，边是关系。
4. 将实体和关系的描述向量化存入向量库，同时保留图结构。

查询时，不仅检索文本段落，还检索相关的实体和关系。这意味着 Agent 可以发现跨文档的间接关联——例如"A公司通过B公司间接投资了C公司"这种需要多跳推理的关系。

GraphRAG 与向量检索是**互补关系**：向量检索擅长找语义相似的文本段（"找相似的"），GraphRAG 擅长发现实体间的间接关系链（"找关联的"）。
Agent 的知识库工具默认使用 `auto`，也可显式选择 `graph`；关系查询会并行执行
GraphRAG 和基础混合检索，再统一排序。图谱更新在私有副本上完成，查询继续读取
旧的一致快照，完成后再短时间原子替换。大文档按字符预算从全文均匀取样，未变化
社区按实体内容指纹复用摘要，并应用最小社区规模与实体/摘要模型配置。
文档禁用状态由 PostgreSQL 目录统一管理：向量检索、BM25 和 GraphRAG 查询都会
排除禁用来源；共享图谱实体只使用仍启用文档的贡献生成查询结果。禁用不会删除
任何索引，重新启用后可立即参与检索。

**⚠️ 曾踩过的坑**：GraphRAG 的 `Community` dataclass 定义了 `id`、`entities`、`summary` 三个字段，但 `save()` 方法访问了 `c.entity_ids` 和 `c.label`，`load()` 又用 `entity_ids=set(...)` 构造——字段不同步导致序列化错误。教训：数据模型变更后，所有序列化/反序列化代码必须同步更新。

---

### 3.3 文档处理管线（ingestion/）

文档处理管线负责将用户上传的各种格式文档转换为可检索的向量索引。

#### 3.3.1 多格式解析（parsers.py）

支持 PDF、DOCX、HTML、Markdown、TXT 五种文档格式的解析。每种格式有不同的处理策略：

- **PDF**：小文件串行解析；大文件默认使用 `spawn` 多进程按页范围解析，
  worker 数、执行器和并行阈值均由 `.env` 配置，并保留线程回退。扫描件仍需要
  额外 OCR 支持。
- **DOCX**：提取带格式的 Word 文档内容，保留标题样式（用于后续的层次化索引）。
- **HTML**：去除标签和样式，提取正文内容。
- **Markdown / TXT**：直接提取纯文本。

解析器通过 `.env` 对上传体积、PDF 页数、解析字符数、DOCX 解压规模和最终
Chunk 数建立硬边界。超限属于客户端输入问题，API 返回带实际值和配置上限的
4xx，而不是笼统的 500。当前默认 `API_MAX_PDF_PAGES=600`；大体量 PDF 已覆盖
异步索引、进度和取消路径。

解析完成后使用解析内容 SHA-256 生成稳定文档 ID，不再把任务临时文件名或
mtime 纳入标识。索引签名覆盖分块、Embedding，以及 RAPTOR/GraphRAG 的有效
Provider、模型和可用状态；
相同内容只有在 PostgreSQL、Qdrant 和 BM25 三方完整性一致时才复用。

#### 3.3.2 文本分块（chunker.py）

将长文档分割成适合检索的文本片段（Chunk）。采用**递归分割 + 语义分割**的组合策略：

1. 优先在段落边界断开。
2. 如果段落太长（超过 1024 token），递归地按句子边界切割。
3. 如果句子仍然太长，按固定长度切割（保留上下文重叠，避免切断关键信息）。
4. 控制块大小在 512-1024 token 之间。

Chunk 大小的选择是权衡的艺术：太小（<256 token）则上下文不足，LLM 难以理解完整含义；太大（>2048 token）则噪声多，检索精度下降，且浪费 Token。

#### 3.3.3 Embedding（embedder.py）

将文本块转换为固定维度的向量表示。MindForge 使用 **BGE-M3**（BAAI 开源的多语言 Embedding 模型，1024 维）作为主力模型。

BGE-M3 的特点：
- 支持 100+ 种语言（适合多语言知识库场景）。
- 输入长度可达 8192 token（支持长文本直接编码）。
- 同时支持稠密向量和稀疏向量（与 BM25 互补）。
- 在 MTEB 中文 Benchmark 上表现优异。

Embedding 后端由 `.env` 显式指定为 BGE 或 OpenAI。显式后端初始化失败时，
索引会报错并停止，不再静默写入 hash 向量，因为不同向量空间混入同一
Collection 会造成不可恢复的检索污染。仅在未指定 provider 的开发模式下保留
hash fallback。

#### 3.3.4 RAPTOR 索引构建（raptor.py）

在 ingestion 阶段，RAPTOR 模块负责将已分块的文档递归聚类生成摘要树。这是一个离线过程，在文档上传后自动触发。

---

### 3.4 工具系统（tools/）

Agent 通过工具与外部世界交互。每个工具遵循统一的规范：名称（name）、描述（description）、参数（JSON Schema）和处理函数（handler）。

#### 3.4.1 RAGTool（rag_tool.py）

Agent 检索内部知识库的主要工具。内部调用混合检索管线（向量 + BM25 + CrossEncoder），返回与查询最相关的文档片段。

#### 3.4.2 WebSearch（web_search.py）

允许 Agent 搜索互联网获取实时信息。`WebSearchTool` 按以下顺序选择后端：

1. 当前模型供应商的原生联网协议。
2. 已配置的 Tavily。
3. 显式启用的 DuckDuckGo HTML 搜索。

OpenAI 与 DeepSeek 使用 Responses Web Search；Kimi 使用 `$web_search` 内置工具；
GLM 使用结构化 Web Search API。兼容云 API 和本地服务通过
`LLM_*_NATIVE_WEB_SEARCH_PROTOCOL` 选择协议，不再把供应商名称硬编码进 Agent。
Tavily 和 DuckDuckGo 都是可选辅助能力，未配置时不会阻止模型正常回答。

#### 3.4.3 CodeExecutor（code_executor.py）

Agent 可以在隔离子进程中执行 Python 代码。沙箱限制 CPU、地址空间、代码/变量/输出体积，并通过静态 AST、导入白名单和审计 Hook 拒绝文件修改、进程、网络和动态库操作。`numpy`、`pandas` 等静态声明的白名单科学库会在审计 Hook 安装前完成可信初始化，之后用户代码仍不能新增动态库加载。

#### 3.4.4 CitationVerifier（citation_verifier.py）

负责检查 `[N]` 引用编号、来源是否存在、来源是否为空，以及声明与来源标题/
正文是否具备最低限度的词汇支持。它是保守的一致性检查器，不等同于事实核查器，
也不声明未经基准验证的幻觉率指标。

#### 3.4.5 MCP 历史说明

旧 `tools/mcp_adapter.py`、`src/mindforge/mcp/`、演示脚本和隔离测试已从当前
`main` 工作树移除。JSON-RPC、stdio 子进程与工具发现说明仅保留在学习文档，
不属于可导入模块或部署能力。若未来重新启用，必须作为独立方案重新完成权限、
生命周期、并发、超时和端到端测试评审。

---

### 3.6 模型适配层（models/）

将所有 LLM 调用抽象为统一接口，支持 **一键切换 LLM 供应商**。

#### 3.6.1 设计动机

不把 LLM 调用硬编码到 Agent 逻辑中，而是通过适配器模式实现供应商切换。这样：
- Agent 代码不关心用的是哪个模型。
- 可以在 OpenAI、DeepSeek、Kimi、GLM、通用兼容 API 和服务器本地模型之间切换。
- 新增原生供应商只需注册 `ProviderBuilder`，不需要修改 Agent 代码。

#### 3.6.2 统一接口

所有适配器实现 `BaseLLM`：
- `chat(messages, tools, response_format, stream)`：统一普通、流式、工具调用和
  结构化输出。
- `search_web(query, max_results, max_output_tokens)`：统一供应商原生联网搜索，
  返回规范化正文、来源、Token usage、模型和后端标识。

原生联网协议由独立适配器实现，目前包括 `openai_responses`、`kimi_builtin` 和
`glm_web_search`。聊天接口、模型列表和四 Agent 路由仍复用统一 Provider 适配层。
- `embed(texts)` / `embed_single(text)`：可选的统一 Embedding 接口。
- 返回 `ChatResult` / `StreamEvent`，Agent 不接触供应商 SDK 响应对象。

#### 3.6.3 注册表与 OpenAI-compatible 适配器

`LLMFactory` 内置 `openai`、`deepseek`、`kimi`、`glm`、
`openai_compatible`、`local` 六个 Provider，并提供注册/注销接口。
Kimi、GLM 和通用接口拥有独立的 Base URL、API Key、模型路由与联网协议配置，
实现层复用 `OpenAICompatibleAdapter`。该适配器统一普通 Chat、
流式响应、Tool Calling 增量聚合、JSON Mode/Schema 能力降级和可选 Embedding。
普通和流式响应都会归一化 Provider usage；流式综合通过最终 usage 事件回传 Token，
兼容端点可通过 `.env` 声明是否支持 `stream_options.include_usage`。

OpenAI 与 DeepSeek 保留原有 Adapter 和导入路径；Kimi 与 GLM 使用独立
Provider 配置；通用接口可连接其他提供 OpenAI-compatible 协议的服务；Local
Provider 可连接 vLLM、Ollama、LM Studio。四个 Agent 角色可独立覆盖模型名，
未覆盖时使用 Provider 默认模型。

本地模型可不配置 API Key，但 Base URL 与模型名必须完整。工具调用、JSON Mode
和 JSON Schema 按 Provider 显式声明；关闭后 Adapter 不会发送服务不支持的参数。
Docker Compose 配置 `host.docker.internal:host-gateway`，应用容器可访问宿主机
推理端点。

模型价格不再写死在代码中。`LLM_MODEL_PRICING` 按 Provider 和模型配置每百万
Token 的输入、缓存输入与输出价格；系统只显示估算费用，并明确区分价格未配置、
usage 缺失、本地模型和部分估算，未知情况不会显示成零费用。

#### 3.6.4 配置与验证流程

1. 打开 `/settings`，选择 OpenAI、DeepSeek、Kimi、GLM、通用接口或本地模型。
2. 配置 Base URL、API Key，检测连接并拉取模型，再选择主要或默认模型。
3. 需要角色分流时展开高级模型路由；协议与能力开关位于高级接口设置。
4. 保存后确认 Provider 状态为“可用”，并检查 `/api/v1/settings` 中
   `llm_configured=true`。
5. 本地模型从应用容器通过 `host.docker.internal` 访问服务器宿主机；服务必须
   监听可被 Docker 网桥访问的地址。
6. RAPTOR、GraphRAG 和 QA 专用模型留空时继承当前 Researcher 模型。

研究 SSE 连接由应用级会话管理，不跟随研究页组件卸载。站内路由切换不会取消
任务，返回研究页后可继续查看进度并停止；浏览器刷新或关闭时发送 keepalive
取消请求并释放连接。

完整命令、Ollama/vLLM 示例和故障排查见
[LLM Provider 配置与运维](llm-provider-operations.md)。

**⚠️ 曾踩过的坑**：API Key 配置的全链路一致性——前端保存 Key 后，后端需要同步更新到环境变量、数据库、缓存三处，任一环节断链都会导致调用失败。脱敏 Key 回显后又被当作真实 Key 写回的问题也花了不少时间排查。

---

### 3.7 记忆系统（memory/）

记忆系统让 MindForge 具备"经验积累"能力，采用**三层记忆架构**，类比人类记忆机制。

#### 3.7.1 工作记忆（working.py）

存储**当前研究任务**的上下文——相关语义记忆、Planner 推理、中间子任务结果和来源片段。传给单个 Researcher 的共享上下文受 `AGENT_RESEARCH_CONTEXT_MAX_CHARS` 限制；前置子任务结果通过依赖上下文单独传递，避免同一结果重复注入。容量超限时按重要性与新鲜度淘汰，会话结束后释放。

#### 3.7.2 情节记忆（episodic.py）

保存完整成功结果及来源、质量、费用、Token 和模型元数据。只有完全相同且未过期的
任务可以直接命中缓存；Planner 回退、Critic 失败、引用无效、精炼失败或存在失败
子任务的结果不会写入或复用。缓存命中保留原生成用量供内部追溯，但本次请求的
Token 清零并标记为不产生新的 API 费用。

#### 3.7.3 语义记忆（semantic.py）

语义记忆只写入已评审且达到质量阈值的完整成功报告。召回同时考虑原始研究问题和报告内容，要求最低查询覆盖率与相似度，并限制参与匹配、单条注入和总注入字符数。相关参数通过 `MEMORY_SEMANTIC_*` 环境变量配置。

持久化成功研究结果及其来源和置信度。新任务开始时按关键词、置信度和新鲜度检索相关记录，并加入工作记忆供 Researcher 使用。当前实现使用受容量、保留时间和文件大小限制的本地 JSON 存储。

---

### 3.8 可观测性（observability/）

#### 3.8.1 Langfuse 3 集成

Langfuse 是开源的 LLM 可观测性平台。SDK 是正式运行时依赖，只有公钥、私钥和 Host 同时配置时才启用远程追踪。MindForge 将其用于跟踪：
- 每次研究一个合法的 32 位 Trace ID，REST、SSE、最终结果和历史记录保持一致。
- 每个 LLM 请求的输入、输出、延迟、Token 数、模型名。
- 每个 Agent 步骤的执行链路（哪个 Agent -> 调用了什么工具 -> 结果如何）。
- Token 消耗分布（按 Agent、按工具分类）。

Langfuse 可直接在服务端 `.env` 配置，也可通过“系统配置 → 可观测”页面保存。
页面保存最终写入同一个 `.env`，重新读取时公钥和私钥只返回掩码。

#### 3.8.2 本地 JSONL 追踪

除了 Langfuse，MindForge 还将 Agent 执行日志写入本地 JSONL 文件，并为完成的
根 Trace 写入小型摘要文件。只读 Repository 负责有界扫描、Trace ID 校验和详情
读取，API 与前端不直接依赖 Langfuse SDK。

两者互为补充：Langfuse 提供在线分析和可视化能力，本地文件为应用内“可观测”
页面和离线审计提供数据。内容默认不写入，仅记录类型、大小和受控元数据；前端
接口不返回 Langfuse 凭证、服务端目录或未脱敏内容。Trace 默认永久保留，支持
删除单条或清空本地 Trace。失败节点记录阶段、错误码、异常类型、Agent、模型、
尝试次数和超时值；详情页展示完整因果链，顶部汇总主因，不把上层超时导致的
LLM 取消误报成独立根因。

---

### 3.9 配置系统（config.py）

使用 `pydantic-settings` 实现统一的配置管理，按功能拆分 16 个子配置类：

| 配置类 | 前缀 | 说明 |
|--------|------|------|
| AppConfig | `MINDFORGE_` | 数据、追踪、语义记忆目录和日志级别 |
| APIConfig | `API_` | 监听、CORS、上传与请求边界 |
| LLMConfig | `LLM_` | 供应商、API Key、模型参数 |
| VectorStoreConfig | `VECTOR_` | Qdrant 连接、Collection 参数 |
| RetrievalConfig | `RETRIEVAL_` | Top-K、RRF 参数、策略选择 |
| ChunkingConfig | `CHUNK_` | Chunk 大小和重叠；具体分块策略由索引请求选择 |
| ParserConfig | `PARSER_` | OCR、表格、资产、解析版本与边界 |
| VisualRetrievalConfig | `VISUAL_` | 默认关闭的视觉描述检索 |
| RAPTORConfig | `RAPTOR_` | 层级、摘要模型、节点上限和摘要并发 |
| GraphRAGConfig | `GRAPH_` | 图谱开关、模型、取样预算、存储、实体和社区上限 |
| AgentConfig | `AGENT_` | 迭代、并发、排队、心跳、超时和 Researcher/Synthesizer/Critic 上下文预算 |
| CacheConfig | `CACHE_` | Redis 连接 |
| MemoryConfig | `MEMORY_` | 工作、情节和语义记忆容量与保留时间 |
| ObservabilityConfig | `OBSERVABILITY_` | Langfuse、本地追踪、脱敏和保留策略 |
| SandboxConfig | `SANDBOX_` | 代码长度、输出、内存和模块白名单 |
| QAGenerationConfig | `QA_` | QA 数据生成模型、批量和重试 |

每个子配置有独立的环境变量前缀，可以从 `.env` 文件或系统环境变量加载。新增配置只需添加子类并注册到 `Settings` 主类。

其中 `DATABASE_URL` 是数据库模块导入时的必填项。应用不会再使用内置
PostgreSQL 连接串；本地、CI 和 Docker 都必须显式提供与目标 PostgreSQL
实例一致的连接 URL。

---

### 3.10 API 层（api/）

#### 3.10.1 REST 路由（routes.py）

所有路由挂载在 `/api/v1` 前缀下，当前共 26 个路由方法：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/query` | POST | 提交研究任务；`stream=true` 时返回 SSE |
| `/index` | POST | 按服务器允许的本地路径建立索引 |
| `/upload` | POST | 流式上传并建立向量、BM25、RAPTOR/GraphRAG 索引 |
| `/index-jobs` | POST/GET | 创建持久化异步索引任务、列出任务 |
| `/index-jobs/{job_id}` | GET/DELETE | 查询进度或请求取消任务 |
| `/documents` | GET | 获取文档列表 |
| `/documents/{doc_id}/enabled` | PATCH | 启用或禁用文档参与检索，不删除索引 |
| `/documents/{doc_id}` | DELETE | 删除文档及关联索引 |
| `/documents/{doc_id}/assets` | GET | 列出持久化资产及可访问的视觉资产 URL |
| `/documents/{doc_id}/assets/{asset_id}` | GET | 读取已登记的图片或页面预览，不暴露源文件路径 |
| `/documents/{doc_id}/content` | GET | 按 Chunk 顺序获取完整文档内容 |
| `/settings` | GET/PUT | 系统配置的读写 |
| `/health` | GET | 服务健康检查 |
| `/ready` | GET | 核心依赖就绪检查，失败返回 503 |
| `/stats` | GET | 系统统计数据 |
| `/history` | GET/POST/DELETE | 历史分页、保存和清空 |
| `/history/{history_id}` | GET | 按需获取完整报告 |
| `/history/{entry_id}` | DELETE | 删除单条历史 |
| `/observability/status` | GET | 返回脱敏后的本地/远程追踪状态 |
| `/observability/traces` | GET | 分页、搜索和筛选本地 Trace 摘要 |
| `/observability/traces/{trace_id}` | GET | 获取单个 Trace 的有界观察链 |

#### 3.10.2 SSE 流式端点

研究任务的提交使用 SSE（Server-Sent Events）实现流式推送。前端 POST 研究请求后，后端返回 `StreamingResponse`，通过 SSE 事件序列逐步推送 Agent 的执行进度：

```
trace_started    →  { type, trace_id }
plan_ready       →  { type, plan: ResearchPlan }
subtask_start    →  { type, task_id, description }
subtask_result   →  { type, task_id, result: AgentResult }
synthesizing     →  { type, status: "start" | "done" }
answer_chunk     →  { type, content }
critic_feedback  →  { type, score: CriticScore, round }
refining         →  { type, round }
done             →  { type, result: AgentResult }
[DONE]           →  终止标记
```

`trace_id` 同时附加到后续 SSE 事件和最终 `AgentResult`。同步 `/query` 响应也返回
同一字段，前端保存历史时将其写入 `research_history.trace_id`。

**⚠️ 曾踩过的坑**：早期版本中 `[DONE]` 终止帧比 `done` 事件先到达时，前端在没有 `finalResult` 的情况下将状态置为已完成的"白屏"问题。修复：在 `onComplete` 回调中加守卫，确认 `finalResult` 存在后再更新状态。

#### 3.10.3 数据模型（schemas.py）

使用 Pydantic v2 定义请求/响应模型。利用类型注释自动验证和序列化。v2 版本用 Rust 实现底层验证引擎（pydantic-core），性能比 v1 提升 5-10 倍。

---

### 3.11 数据库层（db.py）

基于 SQLAlchemy ORM，**仅支持 PostgreSQL，不提供 SQLite 回退**。`DATABASE_URL`
必须由根目录 `.env` 或进程环境显式提供，缺失时启动失败，避免开发默认账号进入
部署环境。API Key 使用 Fernet 加密，密钥由根目录 `.env` 中的 `APP_SECRET`
提供；缺失时应用会生成并尝试持久化。

主要存储的数据：
- API Key 的 Fernet 加密副本；根目录 `.env` 仍是重启后的运行时配置来源
- 研究任务历史（问题、结果、执行时间、Token/费用状态和紧凑引用来源）
- 文档元数据（文件名、格式、上传时间、索引状态）
- 持久化索引任务（阶段、进度、耗时、取消状态）、文档索引签名和文档资产
- 知识库统计信息

---

## 四、前端模块详解

### 4.1 技术栈选择

| 技术 | 选型理由 |
|------|---------|
| React 19 | 最新的稳定版，改进的 HMR 和更流畅的并发模式 |
| TypeScript 严格模式 | 零容忍类型错误，所有 `noUnusedLocals/Parameters` 全开 |
| Vite 8 | 基于 ESM 的开发服务器，冷启动秒级，HMR 毫秒级 |
| Tailwind CSS v4 | Utility-First 模式，CSS 不随项目增长，一致性高 |
| TanStack Router | TypeScript 友好的路由，支持类型安全的 URL 参数 |
| TanStack Query | 服务端状态管理，自动缓存和重新请求 |
| Zustand | 轻量级状态管理，选择器模式避免不必要的重渲染 |

### 4.2 页面设计

#### 4.2.1 概览 Dashboard

首页，展示系统关键状态：
- Qdrant / Redis / PostgreSQL 的连接状态（健康/异常/未连接）。
- 知识库统计（文档总数和索引 Chunk 数）。
- 快捷操作入口（"发起新研究"、"上传文档"、"查看历史"）。

设计思路：用户一进来就能快速了解系统"能用不能用"、"有多少知识储备"，而不是看到一个空白的欢迎页。

#### 4.2.2 研究工作台

核心页面，分为三个区域：

1. **输入区**：搜索框 + 提交按钮。支持快捷键提交。当前 LLM Provider 配置不完整时按钮显示“知识库检索”，请求不会初始化 Multi-Agent；问候类输入直接返回模式说明，不触发无关文档召回。
2. **执行可视化区**：使用 React Flow 实时渲染 DAG 执行图——每个节点是一个子任务，边表示依赖关系。已完成/执行中/等待中的节点用不同颜色区分。规划开始前发送 `planning`，长步骤通过 `heartbeat` 保持连接和状态可见。
3. **结果展示区**：Agent 完成后显示结构化 Markdown 报告、总 Token 和估算费用状态。
包含 Critic 雷达图、未评审/评审失败状态、精炼记录和降级原因。最终报告会再次执行
确定性引用校验；编号越界、缺少引用或来源证据不足时结果标记为降级。正文中的有效
`[N]` 通过 Markdown AST 与结构化来源关联：Web 来源在新标签页打开，内部知识库
来源跳到报告底部条目。成功完成后输入框清空，失败时保留问题便于重试。

#### 4.2.3 知识库页面

文档管理界面：
- 文档选择上传，自动触发解析和索引。
- 文档列表（文件名、Chunk 数、索引状态，以及实际成功应用的基础索引、RAPTOR、
  GraphRAG 标识）。
- 文档检索开关；禁用后保留内容和索引，但不参与向量、BM25 或 GraphRAG 检索。
- 文档删除（删除时自动清理对应的向量索引）。
- 上传时可选择 RAPTOR 和 GraphRAG 增强索引。

#### 4.2.4 研究历史页面

自动保存成功完成的研究任务：
- 任务列表（问题摘要、时间、执行时间、是否成功）。
- 可展开预览（使用与研究结果相同的 Markdown、GFM、代码高亮和可点击引用渲染）。
- 删除 / 清空管理（`API_MAX_HISTORY_ENTRIES=0` 表示默认永久保留）。

#### 4.2.5 可观测页面

独立的运维视图，不与研究报告渲染耦合：
- 左侧按状态、Trace ID 或研究问题搜索，列表和详情以实际研究问题为标题。
- 观察链按父子层级展示 Orchestrator、四个 Agent、LLM 和工具调用，并使用耗时
  条显示相对执行位置。
- 失败节点展开后直接显示原因、阶段、错误码、异常类型、Agent、模型和尝试次数；
  顶部自动汇总主要原因，已恢复的重试显示为链路异常而不是整次研究失败。
- 存在未完成子任务、Planner 回退、Critic 失败、精炼失败或最终引用无效时，顶层结果
  和 Trace 标记为 `degraded` 并显示原因；该报告不会进入成功记忆缓存。
- 具备加载、空、错误、降级、截断和移动端状态；配置 Langfuse 后可跳转到远端 Trace。
- 支持删除当前 Trace 或清空本地 Trace，不影响研究历史与 Langfuse 远端数据。
- 研究结果和历史详情通过 `traceId` URL 参数直接定位，前端不调用 Langfuse API。

#### 4.2.6 系统配置页面

允许用户动态调整配置，无需重启服务：
- 四种 LLM Provider 切换；独立配置 Base URL、API Key、默认/角色模型与能力开关。
- 快速、均衡、深度三种研究模式，以及知识库/联网来源策略和失败回退开关。
- 语义相关性阈值、关键词覆盖阈值、Top-K 和 Embedding Provider。
- 单次模型、子任务、研究总超时采用包含关系校验，避免内部超时预算互相冲突。
- Langfuse Host/凭证、内容采集开关、Trace 与历史保留策略。
- 按当前尚未保存的 Base URL 和凭证调用 `/settings/models`，从 Provider 的
  `/models` 返回值生成角色模型下拉框；未枚举模型保留自定义 ID 输入。
- 向量召回 Top-K 与重排 Top-K。
- Agent 最大迭代、精炼轮数、评判阈值和超时。

LLM 供应商与 Embedding 后端相互独立。设置页不会随 LLM 切换自动修改
Embedding；已有索引时后端拒绝直接切换 Embedding provider，必须先清空并重建
知识库。Provider 配置状态由后端综合判断，不再把“存在 API Key”等同于“模型
可用”；Local Provider 在关闭 Key 要求后可无 Key 运行。
保存设置后，前端必须重新读取服务端配置成功才显示保存完成。后端按变更范围重置
Orchestrator、Retriever 或 Embedder，不再为普通检索参数修改关闭 Qdrant 客户端
或重置索引并发器；活动索引任务存在时拒绝切换 Embedding provider。
模型发现不保存草稿、不返回 Key 明文、不跟随重定向，并限制目标网络范围、超时、
响应大小和返回数量；连接草稿变化后会丢弃旧请求结果。

### 4.3 状态管理

使用 Zustand 管理四个领域的状态：

- **research-store**：研究会话状态，管理 SSE 事件处理、子任务状态、当前报告内容。选择器模式确保只有关注该部分的组件会重渲染。
- **ui-store**：UI 状态，包括主题（亮/暗/自动跟随系统）、侧边栏展开/折叠。
- **history-store**：研究历史列表与详情缓存，保存紧凑来源元数据，服务端默认保留 1000 条。
- **settings-store**：持久化非敏感的界面与参数草稿；完整 API Key 不写入浏览器持久化存储。

### 4.4 SSE 流式渲染

前端使用 `eventsource-parser` 库解析 SSE 事件流。收到事件后：

1. `plan_ready`：更新 DAG 图，显示任务分解结果。
2. `subtask_start`：更新对应节点状态为"执行中"（蓝色脉冲动画）。
3. `subtask_result`：更新节点状态为"已完成"（绿色勾），追加内容。
4. `critic_feedback`：更新雷达图显示 5 维评分。
5. `trace_started`：保存本次研究的 Trace ID。
6. `done`：渲染最终报告，连同 Trace ID 更新历史记录。

`done` 事件还携带归一化后的 `token_usage`、`cost_usd`、`cost_status` 和
`data.sources`。历史记录保存 Token、费用状态、Trace ID 与裁剪后的来源元数据，研究页与
历史页复用同一 Markdown 渲染器。来源 URL 在前后端都限制为 `http/https`；
历史列表不返回来源，展开详情时再加载，避免分页响应随来源数量膨胀。

这种逐步渲染的体验让用户感觉"有人在后台为我工作"，而不是面对一个 30 秒的白屏。

**⚠️ 曾踩过的坑**：流式 tool_calls 未增量聚合——OpenAI 的流式 API 把 tool_calls 的参数分多个 chunk 返回，早期代码直接透传产生了碎片化的工具调用。修复：按 `tc.index` 维护累加器，流结束后一次性发出完整的工具调用。

---

## 五、部署与运维

### 5.1 一键启动（start.sh）

`start.sh` 脚本实现从代码到服务的全自动部署：

1. **前置校验**：确认 `.env`、`requirements.lock`、Python、npm 和 Docker Compose 存在。
2. **启动基础设施**：`docker compose up -d qdrant redis postgres`。
3. **锁定依赖安装**：后端使用 `pip --require-hashes`，前端使用 `npm ci`。
4. **构建或启动前端**：生产模式构建静态资源，开发模式启动 Vite。
5. **启动后端**：通过 `--app-dir src` 启动 uvicorn。
6. **就绪检查**：轮询 `/api/v1/ready`，超时或后端退出时返回非零状态。

### 5.2 开发模式

`bash start.sh --dev` 启动开发模式：
- 前端使用 Vite 开发服务器（热重载，端口 5173）。
- 后端使用 uvicorn --reload（代码修改自动重启）。
- Vite 代理将 `/api/*` 请求转发到后端 8000 端口。

### 5.3 Docker 部署

本地和服务器使用同一套 `.env.example` 键，但实际值不一定相同。远程更新时
不能直接用本地 `.env` 覆盖服务器文件，否则可能改变 PostgreSQL 主机端口、
应用监听地址、容器数据目录或 `APP_UID/APP_GID`。正确流程是先备份远端
`.env`，按键合并新增配置，再执行 `docker compose config --quiet`、重建目标
服务并检查 `/api/v1/ready`。切换 UID/GID 时还要迁移应用数据卷和模型缓存
所有权，并确认绑定挂载文件仍可由容器进程读取。

`Dockerfile` 定义了容器的完整构建过程：
- 第一阶段：Node.js 22 构建 React 静态资源。
- 第二阶段：Python 3.11 slim 使用所选 requirements 哈希锁安装后端依赖。
  CPU 使用 `requirements-cpu.lock`，GPU 使用 `requirements-gpu.lock`。
- 运行阶段使用非 root 用户，并配置 `/api/v1/ready` 健康检查。

`docker-compose.yml` 定义了四个服务：Qdrant、Redis、PostgreSQL、MindForge（可选，也可以单独使用宿主机运行）。
GPU 部署叠加 `docker-compose.gpu.yml`。具备 NVIDIA Container Toolkit 的环境
应优先使用标准 GPU runtime；需要显式设备映射时，具体设备和驱动路径只配置在
目标环境 `.env` 中，不写入公开文档。

### 5.4 标准操作流程

首次部署先复制 `.env.example`，按服务器实际环境配置数据库、LLM、端口、绑定
地址及 CPU/GPU 锁文件，然后执行：

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8000/api/v1/ready
```

服务就绪后的界面操作顺序为：

1. 在“系统配置”中保存 LLM Provider，确认状态为“可用”。
2. 在“知识库”上传文档，按需启用 RAPTOR/GraphRAG，等待当前任务完成；暂时不希望
   某篇文档参与回答时关闭其检索开关，无需删除或重新索引。
3. 在“研究工作台”提交问题；LLM 不可用时只执行知识库检索。
4. 在“研究历史”查看或清理报告。

更新代码后使用 `git pull --ff-only`、`docker compose config --quiet` 和
`docker compose up -d --build` 完成重建；使用
`docker compose logs -f --tail=200 mindforge` 排查应用日志。普通停止使用
`docker compose stop` 或 `docker compose down`，不得在未备份时增加 `-v`。
README 的“服务器完整操作流程”是面向使用者的主入口。

### 5.5 CI/CD

GitHub Actions 自动运行：
- **ruff check**：固定检查 Python 语法、未定义名称和致命静态错误，避免 Ruff 版本升级改变 CI 规则集。
- **pytest + coverage**：运行完整单元与回归测试。
- **前端门禁**：Vitest、ESLint、TypeScript 和 Vite 生产构建。
- Qdrant + Redis + PostgreSQL 作为 Service Container。
- Docker Compose 展开配置校验。

**⚠️ 曾踩过的坑**：CI 中需要外部服务（Qdrant、Redis）的测试用 `@pytest.mark.integration` 标记跳过，避免 CI 因依赖不可用而失败。

---

## 六、分支管理策略

### 6.1 当前分支

仓库只维护 `main`：全栈 Web 平台，包含 React 前端、FastAPI 后端和 PostgreSQL。

### 6.2 历史实验

项目早期曾验证 MCP Client/Server、JSON-RPC 和 stdio 工具发现，但相关运行代码
与实验分支均已移除，不属于当前仓库能力。

### 6.3 维护规则

开发、测试、文档和部署只以 `main` 为准。如未来重新引入 MCP，必须新建独立
方案并重新评估权限、安全、生命周期、并发和端到端测试。

---

## 七、验证与维护

| 指标 | 当前结果 | 验证方式 |
|------|----------|----------|
| Python 致命错误检查 | 通过 | `python -m ruff check src tests --select E9,F63,F7,F82` |
| Python 测试 | 全量通过 | `python -m pytest -q` |
| 前端静态检查 | 通过 | `npm run lint` |
| 前端回归测试 | 全量通过 | `npm test` |
| 前端生产构建 | 通过 | `npm run build` |
| 配置完整性 | 根目录 `.env` 为唯一运行时来源 | Pydantic、Vite、Compose 和脚本共用同一套键 |
| 文档处理 | 页级解析、资产生命周期、取消与进度可追踪 | 回归测试与私有解析基准 |
| 相同内容上传 | 仅在索引完整性一致时复用 | 内容 ID、索引签名与三方完整性核验 |

代码质量门禁以 GitHub Actions 和仓库命令的实际输出为准，不在本文维护易过期的静态告警数量。

尚未在仓库中固化可复现的 NDCG、召回率、BLEU/Rouge-L 或幻觉率基准，因此不把历史估算值作为当前项目结论。
当前 npm 镜像不支持审计接口，因此不声明 `npm audit` 为零漏洞。

---

## 八、项目演进方向

1. **检索评测体系**：固化可重复运行的召回率、NDCG 和引用支持率基准。
2. **端侧 Agent**：探索小模型在端侧运行 Agent 的可能性，降低对云端 API 的依赖。
3. **持续学习**：强化记忆系统的学习能力，让 Agent 从历史交互中自动总结规则和最佳实践。
4. **多模态支持**：扩展到图片理解（通过多模态 LLM）和其他非文本格式的检索。
5. **Agent 安全对齐**：随着 Agent 能力增强，建立更完善的安全机制——Prompt 注入防护、行为约束、审计日志。

---

<p align="center">
  <sub>MindForge · 全栈 Multi-Agent RAG 系统 · 2025-2026</sub>
</p>
