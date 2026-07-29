#!/usr/bin/env python3
"""
MindForge QA 测试集生成器
=========================
通过调用 LLM API 批量生成多领域问答对，用于评估检索系统质量。

用法：
  python scripts/generate_qa_dataset.py --domain computer_science --count 1500
  python scripts/generate_qa_dataset.py --all              # 生成全部 6000 条
  python scripts/generate_qa_dataset.py --resume            # 断点续跑

输出目录：data/qa/{domain}_qa.md
"""

import asyncio
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindforge.config import get_settings, resolve_project_path  # noqa: E402
from mindforge.models.base import (  # noqa: E402
    ChatMessage,
    LLMFactory,
    is_llm_configured,
)

# ── 配置 ──────────────────────────────────────────────────────
_SETTINGS = get_settings()
_QA_CONFIG = _SETTINGS.qa_generation
OUTPUT_DIR = resolve_project_path(_QA_CONFIG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API 配置（从环境变量读取，与 MindForge 一致）
PROVIDER = _SETTINGS.llm.llm_provider
MODEL = (
    _QA_CONFIG.model.strip()
    or _SETTINGS.llm.get_model("researcher", PROVIDER)
)

# 每批生成的 QA 数量（控制 token 消耗和稳定性）
BATCH_SIZE = _QA_CONFIG.batch_size

# 并发请求数（控制 API 限流）
CONCURRENCY = _QA_CONFIG.concurrency

# ── 领域配置 ──────────────────────────────────────────────────
DOMAINS = {
    "computer_science": {
        "name": "计算机科学",
        "total": 1500,
        "topics": [
            "Python 编程基础（数据类型、装饰器、生成器、GIL、异步）",
            "数据结构与算法（数组、链表、树、图、排序、DP）",
            "计算机网络（TCP/IP、HTTP、DNS、TLS、WebSocket）",
            "操作系统（进程线程协程、内存管理、文件系统、IO模型）",
            "数据库（MySQL、PostgreSQL、索引、事务、分库分表）",
            "Redis（数据结构、持久化、集群、分布式锁、缓存策略）",
            "Docker & K8s（容器、镜像、编排、Dockerfile、compose）",
            "Git（分支、rebase、merge、冲突解决、CI/CD）",
            "系统设计（微服务、RESTful API、RPC、消息队列）",
            "LLM & AI（Transformer、Attention、Prompt、RAG、Agent）",
            "向量数据库（Qdrant、HNSW、Milvus、Embedding）",
            "FastAPI & Web框架（路由、依赖注入、中间件、WebSocket）",
            "消息队列（RabbitMQ、Kafka、RocketMQ、消息可靠性）",
            "前端基础（React、TypeScript、状态管理、SSR/CSR）",
        ],
    },
    "law": {
        "name": "法律",
        "total": 900,
        "topics": [
            "民法基础（物权、债权、合同、侵权责任）",
            "刑法基础（犯罪构成、刑罚种类、正当防卫）",
            "商法（公司法、合伙企业法、破产法）",
            "知识产权法（著作权、专利、商标、反不正当竞争）",
            "劳动法（劳动合同、社保、工伤、裁员补偿）",
            "诉讼法（民事诉讼、刑事诉讼、行政诉讼、证据规则）",
            "宪法与行政法（基本权利、行政行为、行政复议）",
            "合同法实务（合同成立、效力、违约、解除）",
            "婚姻家庭法（结婚离婚、财产分割、抚养权）",
            "网络安全与数据保护（《个人信息保护法》《数据安全法》）",
        ],
    },
    "biology": {
        "name": "生物学",
        "total": 900,
        "topics": [
            "分子生物学（DNA复制、转录、翻译、基因表达调控）",
            "细胞生物学（细胞结构、细胞周期、凋亡、信号通路）",
            "遗传学（孟德尔遗传、连锁分析、表观遗传、CRISPR）",
            "进化论（自然选择、物种形成、系统发育树）",
            "生物化学（酶动力学、代谢通路、蛋白质结构）",
            "神经科学（神经元、突触、神经递质、脑区功能）",
            "免疫学（先天免疫、适应性免疫、疫苗、抗体）",
            "生态学（生态系统、种群动态、生物多样性）",
            "生物信息学（序列比对、基因组组装、RNA-seq）",
            "微生物学（细菌、病毒、真菌、抗生素抗性）",
        ],
    },
    "chemistry": {
        "name": "化学",
        "total": 900,
        "topics": [
            "有机化学（官能团、反应机理、立体化学、芳香族化合物）",
            "无机化学（元素周期律、配位化学、晶体场理论）",
            "物理化学（热力学、动力学、量子化学基础、电化学）",
            "分析化学（色谱、质谱、光谱、滴定分析）",
            "生物化学交叉（酶催化、药物设计、代谢组学）",
            "高分子化学（聚合反应、聚合物性质、功能材料）",
            "环境化学（污染物迁移、水化学、大气化学）",
            "计算化学（分子模拟、DFT、力场、QSAR）",
            "化学安全（实验室安全、危化品管理、废弃物处理）",
            "纳米化学（纳米材料合成、表征、应用）",
        ],
    },
    "education": {
        "name": "教育学",
        "total": 900,
        "topics": [
            "教育心理学（学习理论、认知发展、动机、记忆）",
            "教学设计（布鲁姆分类、逆向设计、PBL）",
            "教育技术（在线学习、AI教育、自适应学习）",
            "课程理论（课程开发、课程标准、STEM教育）",
            "教育评价（形成性评价、总结性评价、标准化测试）",
            "教育政策（义务教育、高考制度、双减政策）",
            "比较教育（中外教育体系对比、PISA测评）",
            "特殊教育（融合教育、因材施教、干预策略）",
            "教育研究方法（实验设计、问卷调查、数据分析）",
            "教师发展（教学反思、专业成长、课堂管理）",
        ],
    },
    "engineering": {
        "name": "工程学",
        "total": 900,
        "topics": [
            "软件工程（设计模式、重构、测试、CI/CD、代码审查）",
            "数据工程（ETL、数据仓库、Spark、Flink、数据建模）",
            "AI工程（MLOps、模型部署、A/B实验、特征工程）",
            "网络安全（XSS、CSRF、SQL注入、OWASP Top 10）",
            "DevOps（监控告警、日志系统、容量规划、灾备）",
            "云计算（IaaS/PaaS/SaaS、AWS/GCP/Azure、Serverless）",
            "嵌入式系统（RTOS、驱动开发、I2C/SPI、ARM）",
            "通信工程（4G/5G、调制解调、信道编码、OFDM）",
            "自动化控制（PID控制、PLC、SCADA、工业IoT）",
            "项目管理（敏捷开发、Scrum、看板、需求分析）",
        ],
    },
}

# ┌─ 预设 Prompt 模板 ──────────────────────────────────────────
SYSTEM_PROMPT = """你是一个技术文档专家，负责生成高质量的问答对（QA pairs）用于评估检索系统的质量。
你的回答必须是严谨、准确、结构清晰的。

每个问答对的格式必须严格按照以下示例：
### Q{number}
{问题内容}
**A:** {答案内容}

如果你被要求生成 15 个 QA pair，就输出 15 个，每个独占一个 ### Q{} 区块。
不要添加"好的"、"以下是"之类的多余开场白，直接输出 QA 对。"""


def build_batch_prompt(domain_name: str, topic: str,
                       existing_count: int, total_needed: int,
                       question_types: list[str]) -> str:
    """构建一批 QA 的生成提示。"""
    types_desc = "\n".join(f"- {t}" for t in question_types)
    return f"""领域：{domain_name}
主题：{topic}
已生成：{existing_count} / 总需：{total_needed}

请围绕以上主题，生成 {BATCH_SIZE} 个高质量的问答对，覆盖以下问题类型：

{types_desc}

要求：
1. 问题是中文的，答案也是中文的
2. 答案要准确、详细、有技术深度，不少于 50 字
3. 不要简单定义式的回答，要有实质信息量
4. 部分问题可以要求对比、推理、或给出实例
5. 从基础概念到进阶知识都要覆盖"""


QUESTION_TYPES = [
    "事实型（Factual）：问一个具体的定义、概念、事实。如'什么是XX？'",
    "推理型（Reasoning）：需要结合多个知识点推理。如'为什么XX会这样？'",
    "摘要型（Summary）：要求概括一篇文章或技术的核心思想。如'用一段话概括XX。'",
    "对比型（Comparative）：对比两个概念/技术。如'XX和YY有什么区别？'",
    "流程型（Procedural）：问步骤或方法。如'如何实现XX？'",
    "场景型（Scenario）：给一个实际场景，问怎么选型或解决问题。如'在XX场景下该用A还是B？为什么？'",
    "评价型（Evaluation）：评价某个技术的优缺点或适用边界。如'XX有什么局限性？'",
    "原理型（Mechanism）：问底层原理。如'XX的底层实现原理是什么？'",
]


def parse_qa_blocks(text: str, start_number: int) -> list[dict]:
    """从 LLM 返回的文本中解析 QA 块。"""
    blocks = []
    # 匹配 ### Q{number} 开头的块
    pattern = r'###\s*Q(\d+)\s*\n(.*?)\*\*A:\*\*\s*(.*?)(?=\n###\s*Q|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    for num_str, question, answer in matches:
        blocks.append({
            "number": int(num_str) if num_str.isdigit() else start_number + len(blocks),
            "question": question.strip(),
            "answer": answer.strip(),
        })

    return blocks


def format_qa_block(q: dict) -> str:
    """将 QA 字典格式化为 markdown 块。"""
    return f"""### Q{q['number']}
{q['question']}
**A:** {q['answer']}
"""


# ── 核心生成逻辑 ──────────────────────────────────────────────

class QAGenerator:
    """QA 数据集生成器，支持断点续跑。"""

    def __init__(self):
        self.llm = LLMFactory.create(
            PROVIDER,
            MODEL,
            max_retries=_QA_CONFIG.client_max_retries,
            max_tokens=_QA_CONFIG.max_tokens,
        )
        self.semaphore = asyncio.Semaphore(CONCURRENCY)

    def get_progress_file(self, domain_key: str) -> Path:
        return OUTPUT_DIR / f".progress_{domain_key}.json"

    def load_progress(self, domain_key: str) -> dict:
        """加载某个领域的生成进度。"""
        pf = self.get_progress_file(domain_key)
        if pf.exists():
            try:
                return json.loads(pf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                return {"topic_index": 0, "generated": [], "batch_count": 0}
        return {"topic_index": 0, "generated": [], "batch_count": 0}

    def save_progress(self, domain_key: str, progress: dict):
        """保存生成进度。"""
        pf = self.get_progress_file(domain_key)
        pf.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_output_file(self, domain_key: str) -> Path:
        return OUTPUT_DIR / f"{domain_key}_qa.md"

    async def generate_batch(
        self,
        domain_key: str,
        topic: str,
        existing_count: int,
        total_needed: int,
    ) -> list[dict]:
        """调用 LLM 生成一批 QA。"""
        prompt = build_batch_prompt(
            DOMAINS[domain_key]["name"],
            topic,
            existing_count,
            total_needed,
            QUESTION_TYPES,
        )

        async with self.semaphore:
            for attempt in range(_QA_CONFIG.request_attempts):
                try:
                    response = await self.llm.chat(
                        messages=[
                            ChatMessage(
                                role="system",
                                content=SYSTEM_PROMPT,
                            ),
                            ChatMessage(
                                role="user",
                                content=prompt,
                            ),
                        ],
                        temperature=_QA_CONFIG.temperature,
                    )
                    content = response.content or ""
                except Exception as e:
                    if attempt < _QA_CONFIG.request_attempts - 1:
                        wait = 2 ** attempt * _QA_CONFIG.retry_base_seconds
                        print(
                            f"  [!] 重试 {attempt + 1}/"
                            f"{_QA_CONFIG.request_attempts}，等待 {wait:g}s：{e}"
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(
                            f"  [x] 生成失败"
                            f"（{_QA_CONFIG.request_attempts}次尝试）：{e}"
                        )
                        return []
                    continue

                blocks = parse_qa_blocks(content, existing_count + 1)
                valid = [
                    block
                    for block in blocks
                    if (
                        block["question"]
                        and block["answer"]
                        and len(block["answer"]) > 20
                    )
                ]
                if valid:
                    return valid

                if attempt < _QA_CONFIG.request_attempts - 1:
                    wait = 2 ** attempt * _QA_CONFIG.retry_base_seconds
                    print(
                        "  [!] 本批解析出 0 条有效 QA，"
                        f"等待 {wait:g}s 后重试..."
                    )
                    await asyncio.sleep(wait)

        print(
            "  [x] 连续 "
            f"{_QA_CONFIG.request_attempts} 次未解析出有效 QA"
        )
        return []

    async def generate_domain(self, domain_key: str):
        """为一个领域生成全部 QA。"""
        config = DOMAINS[domain_key]
        output_file = self.get_output_file(domain_key)
        progress = self.load_progress(domain_key)

        # 已有数据
        existing_qas: list[dict] = progress.get("generated", [])
        topic_index = progress.get("topic_index", 0)

        print(f"\n{'='*60}")
        print(f"  领域：{config['name']}（{domain_key}）")
        print(f"  目标：{config['total']} 条")
        print(f"  已有：{len(existing_qas)} 条")
        print(f"  当前主题索引：{topic_index}")
        print(f"{'='*60}")

        # 如果已有足够数据，跳过
        if len(existing_qas) >= config["total"]:
            print("  [OK] 已完成，跳过。")
            # 确保输出文件存在
            self._write_domain_file(output_file, config["name"], existing_qas[:config["total"]])
            return

        batch_count = progress.get("batch_count", 0)

        for ti in range(topic_index, len(config["topics"])):
            topic = config["topics"][ti]
            print(f"\n  [主题 {ti+1}/{len(config['topics'])}]：{topic}")

            # 每个主题生成多批直到均摊目标
            topic_target = max(BATCH_SIZE, config["total"] // len(config["topics"]))
            rounds = 0
            max_rounds = max(3, topic_target // BATCH_SIZE + 1)

            while len(existing_qas) < config["total"] and rounds < max_rounds:
                needed = config["total"] - len(existing_qas)
                if needed <= 0:
                    break

                # 计算当前批次的期望编号
                next_number = max(len(existing_qas), 1) + 1

                qas = await self.generate_batch(
                    domain_key, topic, len(existing_qas), config["total"]
                )

                if not qas:
                    rounds += 1
                    continue

                # 重新编号，避免重复
                for i, q in enumerate(qas):
                    q["number"] = next_number + i

                existing_qas.extend(qas)
                rounds += 1
                batch_count += 1

                print(f"    批次 {batch_count}：+{len(qas)} 条 -> 累计 {len(existing_qas)}/{config['total']}")

                # 保存进度
                progress = {
                    "topic_index": ti,
                    "generated": existing_qas,
                    "batch_count": batch_count,
                }
                self.save_progress(domain_key, progress)

                # 实时写入文件（防止中断丢失）
                self._write_domain_file(
                    output_file, config["name"],
                    existing_qas[:config["total"]]
                )

                if len(existing_qas) >= config["total"]:
                    break

                # 避免 API 限流
                await asyncio.sleep(1.5)

            # 更新主题索引
            progress["topic_index"] = ti + 1
            self.save_progress(domain_key, progress)

        # 最终写入
        final_qas = existing_qas[:config["total"]]
        self._write_domain_file(output_file, config["name"], final_qas)
        print(f"\n  [OK] {config['name']} 完成：{len(final_qas)}/{config['total']} 条 -> {output_file}")

    def _write_domain_file(self, filepath: Path, name: str, qas: list[dict]):
        """将 QA 列表写入 markdown 文件。"""
        lines = [f"# {name} QA 测试集\n", f"> 共 {len(qas)} 条问答对\n"]

        # 如需要，添加类型划分
        for q in qas:
            lines.append(format_qa_block(q) + "\n")

        filepath.write_text("".join(lines), encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description="MindForge QA 数据集生成器")
    parser.add_argument("--domain", type=str, default="",
                        help="领域名（computer_science / law / biology / chemistry / education / engineering）")
    parser.add_argument("--count", type=int, default=0,
                        help="生成条数（默认使用领域配置）")
    parser.add_argument("--all", action="store_true",
                        help="生成所有领域（共 6000 条）")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="并发请求数（默认读取 QA_CONCURRENCY）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批生成条数（默认读取 QA_BATCH_SIZE）",
    )
    args = parser.parse_args()

    global CONCURRENCY, BATCH_SIZE
    if args.concurrency is not None:
        if args.concurrency < 1:
            parser.error("--concurrency must be at least 1")
        CONCURRENCY = args.concurrency
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        BATCH_SIZE = args.batch_size

    if not is_llm_configured(PROVIDER):
        print("[x] 当前 LLM Provider 配置不完整，请检查项目根目录 .env")
        sys.exit(1)

    generator = QAGenerator()

    domains_to_run = []
    if args.all:
        domains_to_run = list(DOMAINS.keys())
    elif args.domain:
        if args.domain not in DOMAINS:
            print(f"[x] 未知领域：{args.domain}")
            print(f"   可选：{', '.join(DOMAINS.keys())}")
            sys.exit(1)
        domains_to_run = [args.domain]
    else:
        print("请指定 --domain <名称> 或 --all")
        parser.print_help()
        sys.exit(1)

    # 如果指定了 count，覆盖领域配置
    if args.count > 0 and args.domain:
        DOMAINS[args.domain]["total"] = args.count

    print("[开始生成 QA 数据集]")
    print(f"   模型：{MODEL}")
    print(f"   每批：{BATCH_SIZE} 条")
    print(f"   并发：{CONCURRENCY}")
    print(f"   输出：{OUTPUT_DIR}")

    for dk in domains_to_run:
        await generator.generate_domain(dk)

    print(f"\n{'='*60}")
    print("  全部完成！")
    for dk in domains_to_run:
        fp = generator.get_output_file(dk)
        if fp.exists():
            size = fp.stat().st_size
            lines = fp.read_text(encoding="utf-8").count("### Q")
            print(f"  [OK] {dk}：{lines} 条 / {size/1024:.1f}KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
