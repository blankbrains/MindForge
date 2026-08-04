from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mindforge.agents.base import AgentResult
from mindforge.agents.critic import CriticAgent, CriticScore
from mindforge.agents.orchestrator import Orchestrator
from mindforge.agents.planner import PlannerAgent, ResearchPlan, SubTask
from mindforge.agents.researcher import ResearcherAgent
from mindforge.agents.response_guidance import (
    build_response_guidance,
    classify_response_depth,
)
from mindforge.agents.synthesizer import (
    SynthesizerAgent,
    _SYNTHESIZER_SYSTEM_PROMPT,
)
from mindforge.models.base import ChatResult


def _orchestrator_settings(mode: str = "deep") -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(
            research_mode=mode,
            max_refine_rounds=1,
            research_timeout=10,
            subtask_timeout=5,
            queue_timeout=5,
            sse_heartbeat_seconds=1,
            stream_chunk_size=512,
            research_context_max_chars=12_000,
            max_iterations=3,
        ),
        llm=SimpleNamespace(llm_provider="test"),
        observability=SimpleNamespace(enable_tracing=False),
    )


class _SinglePlanner:
    def __init__(self, *, status: str = "planned") -> None:
        self.status = status

    async def run(self, task: str) -> ResearchPlan:
        return ResearchPlan(
            plan_id="single",
            original_task=task,
            subtasks=[
                SubTask(
                    task_id="t1",
                    description=task,
                    subtopics=[task],
                )
            ],
            planner_status=self.status,
            planner_error=("planner failed" if self.status == "fallback" else None),
        )


class _StaticResearcher:
    def __init__(
        self,
        output: str = "draft",
        *,
        sources: list[dict] | None = None,
    ) -> None:
        self.output = output
        self.sources = sources or []

    async def run(self, task: str, *, context=None) -> AgentResult:
        del task, context
        return AgentResult(
            agent_name="researcher",
            output=self.output,
            data={"sources": self.sources},
        )


class _EmptyRefiner:
    async def synthesize(self, **_kwargs) -> AgentResult:
        return AgentResult(
            agent_name="synthesizer",
            success=False,
            output="",
            data={"failure_reason": "empty_llm_response"},
        )


class _LowScoreCritic:
    async def evaluate(self, **_kwargs) -> CriticScore:
        return CriticScore(overall=6.0, should_refine=True)


class _UnusedCritic:
    async def evaluate(self, **_kwargs) -> CriticScore:
        raise AssertionError("critic should not run for this simple balanced task")


def test_ready_tasks_honor_planner_priority() -> None:
    plan = ResearchPlan(
        plan_id="priority",
        original_task="task",
        subtasks=[
            SubTask(task_id="slow", description="low", priority=8),
            SubTask(task_id="urgent", description="high", priority=1),
            SubTask(task_id="normal", description="normal", priority=5),
        ],
    )

    assert [task.task_id for task in plan.get_ready_tasks()] == [
        "urgent",
        "normal",
        "slow",
    ]


@pytest.mark.asyncio
async def test_researcher_receives_task_type_and_subtopics() -> None:
    agent = object.__new__(ResearcherAgent)
    captured: dict[str, object] = {}

    async def run_loop(task: str, **kwargs) -> AgentResult:
        captured["task"] = task
        captured.update(kwargs)
        return AgentResult(agent_name="researcher", output="ok")

    agent._run_tool_loop = run_loop

    await agent.run(
        "实现并验证排序算法",
        task_type="code",
        subtopics=["边界输入", "复杂度"],
    )

    context = str(captured.get("context") or "")
    assert "code" in context
    assert "边界输入" in context
    assert "复杂度" in context


def test_direct_plan_rejects_compound_multi_intent_task() -> None:
    assert (
        Orchestrator._can_use_direct_plan(
            "分析接口超时的原因、影响和解决方案"
        )
        is False
    )


def test_recommendation_question_uses_one_direct_subtask() -> None:
    task = "我如果要学习Agent内容，推荐的编程语言用那个比较好？"

    assert PlannerAgent._minimum_subtask_count(task) == 1
    assert Orchestrator._can_use_direct_plan(task) is True
    assert classify_response_depth(task) == "standard"
    assert "900-1600" in ResearcherAgent.response_length_guidance(task)


def test_deep_research_keeps_detailed_response_guidance() -> None:
    assert "2500-5000" in ResearcherAgent.response_length_guidance(
        "全面深入分析 Agent 框架的架构、生态、风险和发展趋势"
    )


@pytest.mark.parametrize(
    ("task", "expected_depth", "expected_budget"),
    [
        ("请用三句话简要说明什么是 RAG", "concise", "100-500"),
        ("什么是异步编程？", "focused", "500-1000"),
        ("Python 和 Java 应该怎么选？", "standard", "900-1600"),
        (
            "全面分析 Agent 的架构、风险、评测和部署方案",
            "deep",
            "2500-5000",
        ),
    ],
)
def test_response_guidance_matches_user_intent(
    task: str,
    expected_depth: str,
    expected_budget: str,
) -> None:
    assert classify_response_depth(task) == expected_depth
    assert expected_budget in build_response_guidance(task)


def test_code_guidance_prioritizes_complete_code_over_prose_length() -> None:
    guidance = build_response_guidance(
        "实现一个带重试的异步请求函数",
        task_type="code",
    )

    assert classify_response_depth(
        "实现一个带重试的异步请求函数",
        task_type="code",
    ) == "code"
    assert "完整、可运行" in guidance
    assert "中文字符" not in guidance


def test_comparison_plan_does_not_require_redundant_synthesis_subtask() -> None:
    assert PlannerAgent._minimum_subtask_count(
        "Python 和 Java 有什么区别"
    ) == 2


@pytest.mark.asyncio
async def test_planner_respects_a_two_subtask_limit_for_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = object.__new__(PlannerAgent)
    planner._model_name = "planner-model"
    planner._provider_name = "test"

    async def planned_response(*_args, **_kwargs):
        return SimpleNamespace(
            content=json.dumps(
                {
                    "reasoning": "分别研究两个比较对象。",
                    "subtasks": [
                        {
                            "task_id": "python",
                            "description": "研究 Python 的核心特性",
                            "task_type": "research",
                            "dependencies": [],
                            "priority": 1,
                            "subtopics": ["Python 特性"],
                        },
                        {
                            "task_id": "java",
                            "description": "研究 Java 的核心特性",
                            "task_type": "research",
                            "dependencies": [],
                            "priority": 1,
                            "subtopics": ["Java 特性"],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            usage={},
            model="planner-model",
        )

    planner._chat = planned_response
    monkeypatch.setattr(
        "mindforge.agents.planner.get_settings",
        lambda: SimpleNamespace(agent=SimpleNamespace(max_subtasks=2)),
    )

    plan = await planner.run("Python 和 Java 有什么区别")

    assert plan.planner_status == "planned"
    assert len(plan.subtasks) == 2


def test_planner_output_status_is_not_trusted() -> None:
    plan = ResearchPlan.from_dict(
        {
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "研究目标",
                    "status": "completed",
                    "task_type": "research",
                    "priority": 1,
                    "subtopics": ["目标"],
                }
            ]
        }
    )

    assert plan.subtasks[0].status == "pending"


def test_planner_rejects_invalid_execution_fields() -> None:
    plan = ResearchPlan(
        plan_id="invalid",
        original_task="task",
        subtasks=[
            SubTask(
                task_id="t1",
                description="",
                task_type="unknown",
                priority=99,
                subtopics=["topic"],
            )
        ],
    )

    with pytest.raises(ValueError):
        plan.validate()


def test_planner_quality_respects_one_subtask_configuration() -> None:
    plan = ResearchPlan(
        plan_id="single-comparison",
        original_task="Python 和 Java 有什么区别",
        subtasks=[
            SubTask(
                task_id="t1",
                description="研究并比较 Python 和 Java 的核心差异",
                subtopics=["语言特性", "生态", "适用场景"],
            )
        ],
    )

    assert PlannerAgent._quality_errors(
        "Python 和 Java 有什么区别",
        plan,
        max_subtasks=1,
    ) == []


@pytest.mark.parametrize(
    "task",
    [
        "把下面这段英文翻译成中文：hello world",
        "将这段话改写得更正式：系统运行正常",
        "写一首关于春天的短诗",
        "计算 12 * (8 + 2)",
        "写一个 Python 快速排序函数",
    ],
)
def test_non_factual_tasks_do_not_require_sources(task: str) -> None:
    assert ResearcherAgent.requires_sources(task) is False


def test_factual_task_still_requires_sources() -> None:
    assert ResearcherAgent.requires_sources("解释 Python 协程的调度机制") is True
    assert (
        ResearcherAgent.requires_sources(
            "把机器翻译的发展历史和关键技术梳理一下"
        )
        is True
    )


@pytest.mark.asyncio
async def test_preferred_sources_can_fall_back_to_model_only_success() -> None:
    class DirectLLM:
        _model = "direct"

        def __init__(self) -> None:
            self.messages = []

        async def chat(self, messages, **_kwargs) -> ChatResult:
            self.messages = list(messages)
            return ChatResult(content="没有来源的模型回答")

    llm = DirectLLM()
    result = await ResearcherAgent(llm=llm, tools=[]).run(
        "解释 Python 协程的调度机制",
        max_rounds=1,
    )

    assert result.success is True
    assert result.output == "没有来源的模型回答"
    assert result.data["outcome"] == "success"
    assert result.data["grounding_status"] == "model_only"
    assert result.data["citation_status"] == "unavailable"
    assert result.data["failure_reason"] is None
    assert result.data["source_warning"] == "sources_unavailable"
    fallback_prompt = llm.messages[-1].content
    assert "严格遵守用户指定的篇幅、句数和输出格式" in fallback_prompt
    assert "不要在正文中追加来源免责声明" in fallback_prompt
    assert "明确说明本次回答没有经过外部来源核验" not in fallback_prompt


@pytest.mark.asyncio
async def test_required_sources_still_degrade_without_evidence() -> None:
    class DirectLLM:
        _model = "direct"

        async def chat(self, *_args, **_kwargs) -> ChatResult:
            return ChatResult(content="没有来源的最新版本回答")

    task = "请联网核对 Python 最新稳定版本并提供可点击官方来源"
    assert ResearcherAgent.source_requirement(task) == "required"

    result = await ResearcherAgent(llm=DirectLLM(), tools=[]).run(
        task,
        max_rounds=1,
    )

    assert result.success is True
    assert result.data["outcome"] == "degraded"
    assert result.data["grounding_status"] == "model_only"
    assert result.data["failure_reason"] == "sources_unavailable"


@pytest.mark.asyncio
async def test_critic_preserves_severe_issue_refinement_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"

    async def response(*_args, **_kwargs):
        return SimpleNamespace(
            content=json.dumps(
                {
                    "scores": {
                        "completeness": 8,
                        "accuracy": 8,
                        "depth": 8,
                        "clarity": 8,
                        "citations": 8,
                        "overall": 8,
                    },
                    "issues": ["存在严重事实矛盾"],
                    "suggestions": ["修正矛盾"],
                    "should_refine": True,
                },
                ensure_ascii=False,
            ),
            usage={},
            model="critic-model",
        )

    critic._chat = response
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(
                critic_threshold=7.0,
                critic_source_context_max_chars=4_000,
            )
        ),
    )

    score = await critic.evaluate(task="问题", draft="报告")

    assert score.should_refine is True


@pytest.mark.asyncio
async def test_critic_receives_bounded_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"
    captured: dict[str, str] = {}

    async def response(messages, **_kwargs):
        captured["prompt"] = messages[-1].content
        return SimpleNamespace(
            content=json.dumps(
                {
                    "scores": {
                        "completeness": 8,
                        "accuracy": 8,
                        "depth": 8,
                        "clarity": 8,
                        "citations": 8,
                        "overall": 8,
                    },
                    "issues": [],
                    "suggestions": [],
                    "should_refine": False,
                }
            ),
            usage={},
            model="critic-model",
        )

    critic._chat = response
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(
                critic_threshold=7.0,
                critic_source_context_max_chars=200,
            )
        ),
    )

    await critic.evaluate(
        task="问题",
        draft="报告 [7]",
        sources=[
            {
                "index": 7,
                "title": "证据来源",
                "url": "https://example.com",
                "content": "这是用于核对报告事实的证据正文。",
            }
        ],
    )

    assert "[7]" in captured["prompt"]
    assert "这是用于核对报告事实的证据正文" in captured["prompt"]


@pytest.mark.asyncio
async def test_critic_bounds_report_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critic = object.__new__(CriticAgent)
    critic._model_name = "critic-model"
    critic._provider_name = "test"
    captured: dict[str, str] = {}

    async def response(messages, **_kwargs):
        captured["prompt"] = messages[-1].content
        return SimpleNamespace(
            content=json.dumps(
                {
                    "scores": {
                        "completeness": 8,
                        "accuracy": 8,
                        "depth": 8,
                        "clarity": 8,
                        "citations": 8,
                        "overall": 8,
                    },
                    "issues": [],
                    "suggestions": [],
                    "should_refine": False,
                }
            ),
            usage={},
            model="critic-model",
        )

    critic._chat = response
    monkeypatch.setattr(
        "mindforge.agents.critic.get_settings",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(
                critic_threshold=7.0,
                critic_source_context_max_chars=200,
                critic_report_context_max_chars=300,
            )
        ),
    )

    await critic.evaluate(task="问题", draft="报告正文" * 5_000)

    assert len(captured["prompt"]) < 1_500


def test_synthesizer_does_not_fill_missing_evidence_from_model_memory() -> None:
    assert "利用你自己的广博训练知识" not in _SYNTHESIZER_SYSTEM_PROMPT
    assert "缺少证据" in _SYNTHESIZER_SYSTEM_PROMPT
    assert "不得用模型记忆补造事实" in _SYNTHESIZER_SYSTEM_PROMPT
    assert "报告必须按以下结构" not in _SYNTHESIZER_SYSTEM_PROMPT
    assert "不得机械套用固定报告模板" in _SYNTHESIZER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_synthesizer_uses_dynamic_final_report_guidance() -> None:
    class LLM:
        _model = "synth"

    agent = SynthesizerAgent(llm=LLM())
    captured: dict[str, str] = {}

    async def response(messages, **_kwargs):
        captured["prompt"] = messages[-1].content
        return ChatResult(content="综合报告")

    agent._chat = response

    await agent.synthesize(
        task="Python 和 Java 应该怎么选？",
        subtask_results=[
            {"task_id": "python", "output": "Python finding"},
            {"task_id": "java", "output": "Java finding"},
        ],
    )

    assert "1500-2800" in captured["prompt"]
    assert "按实际内容选择章节" in captured["prompt"]
    assert "不得编造引用" in captured["prompt"]


@pytest.mark.asyncio
async def test_synthesizer_bounds_combined_subtask_context() -> None:
    class LLM:
        _model = "synth"

    agent = SynthesizerAgent(llm=LLM())
    agent._settings = SimpleNamespace(
        agent=SimpleNamespace(synthesis_context_max_chars=400)
    )
    captured: dict[str, str] = {}

    async def response(messages, **_kwargs):
        captured["prompt"] = messages[-1].content
        return ChatResult(content="综合报告")

    agent._chat = response

    await agent.synthesize(
        task="综合问题",
        subtask_results=[
            {
                "task_id": "a",
                "description": "A",
                "output": "A" * 5_000,
            },
            {
                "task_id": "b",
                "description": "B",
                "output": "B" * 5_000,
            },
        ],
    )

    assert len(captured["prompt"]) < 2_000


@pytest.mark.asyncio
async def test_empty_refinement_is_reported_as_degraded() -> None:
    orchestrator = Orchestrator(
        planner=_SinglePlanner(),
        researcher=_StaticResearcher(),
        synthesizer=_EmptyRefiner(),
        critic=_LowScoreCritic(),
    )
    orchestrator._settings = _orchestrator_settings()

    result = await orchestrator.run("深入研究")

    assert result.output == "draft"
    assert result.metadata["outcome"] == "degraded"
    assert result.metadata["refinement_status"] == "failed"
    assert result.data["refinement_failure"]


@pytest.mark.asyncio
async def test_stream_empty_refinement_is_reported_as_degraded() -> None:
    orchestrator = Orchestrator(
        planner=_SinglePlanner(),
        researcher=_StaticResearcher(),
        synthesizer=_EmptyRefiner(),
        critic=_LowScoreCritic(),
    )
    orchestrator._settings = _orchestrator_settings()

    events = [event async for event in orchestrator.stream_run("深入研究")]
    result = next(
        event["result"] for event in events if event["type"] == "done"
    )

    assert result.metadata["outcome"] == "degraded"
    assert result.metadata["refinement_status"] == "failed"
    assert result.data["refinement_failure"]


def test_planner_and_critic_failures_degrade_final_outcome() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test")
    )
    plan = ResearchPlan(
        plan_id="fallback",
        original_task="task",
        subtasks=[SubTask(task_id="t1", description="task")],
        planner_status="fallback",
        planner_error="planner failed",
    )
    critic = CriticScore(
        evaluation_status="failed",
        evaluation_error="critic failed",
    )

    result = orchestrator._build_success_result(
        output="report",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "t1",
                "description": "task",
                "success": True,
                "output": "report",
                "sources": [],
            }
        ],
        sources=[],
        final_critic=critic,
        refine_count=0,
        total_usage={},
        elapsed_ms=1,
        total_cost_usd=None,
        cost_status="usage_unavailable",
    )

    assert result.metadata["outcome"] == "degraded"
    assert "Planner" in result.metadata["failure_reason"]
    assert "评审" in result.metadata["failure_reason"]


def test_model_only_subtask_degrades_final_outcome() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test")
    )
    plan = ResearchPlan(
        plan_id="model-only",
        original_task="task",
        subtasks=[SubTask(task_id="t1", description="task")],
    )

    result = orchestrator._build_success_result(
        output="模型回答",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "t1",
                "description": "task",
                "success": True,
                "outcome": "degraded",
                "grounding_status": "model_only",
                "failure_reason": "sources_unavailable",
                "output": "模型回答",
                "sources": [],
            }
        ],
        sources=[],
        final_critic=None,
        refine_count=0,
        total_usage={},
        elapsed_ms=1,
        total_cost_usd=None,
        cost_status="usage_unavailable",
    )

    assert result.success is True
    assert result.metadata["outcome"] == "degraded"
    assert result.metadata["grounding_status"] == "model_only"
    assert "未获得可核验来源" in result.metadata["failure_reason"]


def test_preferred_model_only_subtask_keeps_success_outcome() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._settings = SimpleNamespace(
        llm=SimpleNamespace(llm_provider="test")
    )
    plan = ResearchPlan(
        plan_id="model-only",
        original_task="task",
        subtasks=[SubTask(task_id="t1", description="task")],
    )

    result = orchestrator._build_success_result(
        output="模型回答",
        plan=plan,
        subtask_outputs=[
            {
                "task_id": "t1",
                "description": "task",
                "success": True,
                "outcome": "success",
                "grounding_status": "model_only",
                "failure_reason": None,
                "source_warning": "web_search:native_timeout",
                "output": "模型回答",
                "sources": [],
            }
        ],
        sources=[],
        final_critic=None,
        refine_count=0,
        total_usage={},
        elapsed_ms=1,
        total_cost_usd=None,
        cost_status="usage_unavailable",
    )

    assert result.success is True
    assert result.metadata["outcome"] == "success"
    assert result.metadata["grounding_status"] == "model_only"
    assert result.metadata["source_warning"] == "web_search:native_timeout"
    assert "failure_reason" not in result.metadata


@pytest.mark.asyncio
async def test_evaluation_failed_cache_entry_is_not_reused() -> None:
    cached = AgentResult(
        agent_name="orchestrator",
        output="未经有效评审的报告",
        data={"critic_score": None, "sources": []},
        metadata={
            "quality": None,
            "quality_status": "evaluation_failed",
            "outcome": "success",
        },
    )

    class Memory:
        async def recall(self, _task: str):
            return cached.to_dict()

    orchestrator = object.__new__(Orchestrator)
    orchestrator._episodic_memory = Memory()

    result = await orchestrator._recall_cached_result(
        "你好",
        start_time=0,
    )

    assert result is None


@pytest.mark.asyncio
async def test_stale_native_search_cache_is_not_reused() -> None:
    cached = AgentResult(
        agent_name="orchestrator",
        output=(
            "[Python Releases]([2]windows/) and "
            "https://example.com/#ws_call_id=old"
        ),
        data={
            "sources": [
                {
                    "index": 1,
                    "title": "Old source",
                    "url": (
                        "https://example.com/"
                        "#ws_call_id=old"
                    ),
                }
            ]
        },
        metadata={"outcome": "success"},
    )

    class Memory:
        async def recall(self, _task: str):
            return cached.to_dict()

    orchestrator = object.__new__(Orchestrator)
    orchestrator._episodic_memory = Memory()

    result = await orchestrator._recall_cached_result(
        "cached task",
        start_time=0,
    )

    assert result is None


@pytest.mark.asyncio
async def test_invalid_final_citation_marks_report_degraded() -> None:
    orchestrator = Orchestrator(
        planner=_SinglePlanner(),
        researcher=_StaticResearcher(
            "协程是一种并发抽象 [2]。",
            sources=[
                {
                    "index": 1,
                    "title": "协程资料",
                    "url": "https://example.com/coroutine",
                    "content": "协程是一种并发抽象。",
                }
            ],
        ),
        synthesizer=SimpleNamespace(),
        critic=_UnusedCritic(),
    )
    orchestrator._settings = _orchestrator_settings(mode="balanced")

    result = await orchestrator.run("什么是协程")

    assert result.success is True
    assert result.metadata["outcome"] == "degraded"
    assert result.data["citation_verification"]["valid"] is False
    assert result.metadata["citation_status"] == "invalid"
