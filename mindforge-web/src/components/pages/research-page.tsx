import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useResearchSession } from "@/hooks/use-research-session";
import { useResearchStore } from "@/store/research-store";
import { useSettingsStore } from "@/store/settings-store";
import { Link } from "@tanstack/react-router";
import { QueryInput } from "@/components/research/query-input";
import { SubtaskProgressList } from "@/components/research/subtask-progress-list";
import { EmptyState } from "@/components/shared/empty-state";
import { LoadingSkeleton } from "@/components/shared/loading-skeleton";
import {
  Activity,
  AlertTriangle,
  Clock3,
  FileSearch,
  KeyRound,
  ListChecks,
  Loader2,
  Search,
  XCircle,
} from "lucide-react";

const PlanDAG = lazy(() =>
  import("@/components/research/plan-dag").then((module) => ({
    default: module.PlanDAG,
  })),
);
const CriticFeedbackPanel = lazy(() =>
  import("@/components/research/critic-feedback-panel").then(
    (module) => ({ default: module.CriticFeedbackPanel }),
  ),
);
const ReportViewer = lazy(() =>
  import("@/components/research/report-viewer").then((module) => ({
    default: module.ReportViewer,
  })),
);
const StreamingAnswerPanel = lazy(() =>
  import("@/components/research/streaming-markdown").then(
    (module) => ({ default: module.StreamingAnswerPanel }),
  ),
);

export function ResearchPage() {
  const session = useResearchSession();
  const task = useResearchStore((s) => s.task);
  const setTask = useResearchStore((s) => s.setTask);
  const hasLLMKey = useSettingsStore((s) => s.hasLLMKey);
  const lastTaskRef = useRef(task);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!session.isStreaming || !session.startedAt) return;
    const interval = window.setInterval(() => {
      setElapsedSeconds(
        Math.max(0, Math.floor((Date.now() - session.startedAt!) / 1000)),
      );
    }, 1000);
    return () => window.clearInterval(interval);
  }, [session.isStreaming, session.startedAt]);

  const taskStats = useMemo(() => {
    const subtasks = Object.values(session.subtasks);
    return {
      total: subtasks.length,
      completed: subtasks.filter((item) => item.status === "completed").length,
      failed: subtasks.filter((item) => item.status === "failed").length,
    };
  }, [session.subtasks]);
  const phaseLabel: Record<string, string> = {
    connecting: "正在连接",
    starting: "正在启动",
    planning: "正在规划",
    researching: "正在执行子任务",
    synthesizing: "正在合成报告",
    reviewing: "正在质量评审",
    refining: "正在精炼报告",
    completed: "研究完成",
    failed: "研究失败",
  };

  const handleSubmit = useCallback(
    (t: string) => {
      lastTaskRef.current = t;
      setElapsedSeconds(0);
      session.startResearch(t);
    },
    [session],
  );

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">研究工作台</h1>
        <p className="mt-1 text-text-muted">输入研究问题，观察 Multi-Agent 系统实时协作</p>
      </div>

      {/* 当前 Provider 未就绪时进入知识库检索模式。 */}
      {!hasLLMKey && (
        <div className="flex items-center gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>当前模型服务配置不完整，将使用<strong>文档检索模式</strong>。如需 AI 分析，请先</span>
          <Link to="/settings" search={{}} className="font-medium underline whitespace-nowrap">配置模型</Link>
        </div>
      )}

      <QueryInput
        value={task}
        onChange={setTask}
        onSubmit={handleSubmit}
        isRunning={session.isStreaming}
        onCancel={session.cancelResearch}
        retrievalOnly={!hasLLMKey}
      />

      {session.isStreaming && (
        <section
          aria-label="研究执行状态"
          className="grid grid-cols-1 gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-3"
        >
          <div className="flex items-center gap-3 bg-surface px-4 py-3">
            <Activity className="h-4 w-4 text-primary" />
            <div>
              <p className="text-xs text-text-muted">当前阶段</p>
              <p className="text-sm font-semibold">
                {phaseLabel[session.phase] ?? "处理中"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-surface px-4 py-3">
            <Clock3 className="h-4 w-4 text-text-muted" />
            <div>
              <p className="text-xs text-text-muted">已用时间</p>
              <p className="text-sm font-semibold">
                {Math.floor(elapsedSeconds / 60)} 分{" "}
                {elapsedSeconds % 60} 秒
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 bg-surface px-4 py-3">
            <ListChecks className="h-4 w-4 text-text-muted" />
            <div>
              <p className="text-xs text-text-muted">子任务</p>
              <p className="text-sm font-semibold">
                {taskStats.total
                  ? `${taskStats.completed}/${taskStats.total} 完成`
                  : "等待规划"}
                {taskStats.failed ? ` · ${taskStats.failed} 失败` : ""}
              </p>
            </div>
          </div>
        </section>
      )}

      {session.isError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          <XCircle className="h-5 w-5 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <p>{session.error ?? "未知错误"}</p>
            <div className="flex gap-2">
              {session.traceId && (
                <Link
                  to="/observability"
                  search={{ traceId: session.traceId }}
                  className="inline-flex items-center gap-1 rounded-md bg-red-100 px-3 py-1.5 text-xs font-medium hover:bg-red-200 dark:bg-red-900 dark:hover:bg-red-800"
                >
                  <Activity className="h-3 w-3" />
                  查看失败链路
                </Link>
              )}
              <Link to="/settings" search={{}} className="inline-flex items-center gap-1 rounded-md bg-red-100 px-3 py-1.5 text-xs font-medium hover:bg-red-200 dark:bg-red-900 dark:hover:bg-red-800">
                <KeyRound className="h-3 w-3" />检查模型配置
              </Link>
              <button type="button" onClick={() => session.startResearch(lastTaskRef.current)} className="inline-flex items-center gap-1 rounded-md bg-amber-100 px-3 py-1.5 text-xs font-medium hover:bg-amber-200 dark:bg-amber-900 dark:hover:bg-amber-800">
                <FileSearch className="h-3 w-3" />重试
              </button>
            </div>
          </div>
        </div>
      )}

      {session.isIdle && !session.finalResult && (
        <EmptyState icon={<Search className="h-12 w-12" />} title="开始新的研究" description="输入一个问题，Agent 将自动分解任务、检索信息、生成报告" />
      )}

      {session.isStreaming && (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="xl:col-span-2 space-y-6">
            {/* 流式答案：逐字渲染，在报告完成前就给用户看到内容 */}
            <Suspense fallback={<LoadingSkeleton variant="text" count={4} />}>
              <StreamingAnswerPanel />
            </Suspense>
            <div>
              <h4 className="mb-2 text-sm font-semibold">
                任务 DAG
                {taskStats.total
                  ? ` · ${taskStats.total} 个子问题 · ${taskStats.completed}/${taskStats.total} 完成`
                  : ""}
              </h4>
              <Suspense fallback={<LoadingSkeleton variant="card" count={1} />}>
                <PlanDAG plan={session.plan} />
              </Suspense>
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <SubtaskProgressList subtasks={session.subtasks} />
              <Suspense fallback={<LoadingSkeleton variant="card" count={1} />}>
                <CriticFeedbackPanel score={session.criticScore} />
              </Suspense>
            </div>
            {session.refineRound > 0 && (
              <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300">
                <Loader2 className="h-4 w-4 animate-spin" /> 精炼中… 第 {session.refineRound} 轮
              </div>
            )}
          </div>
          <div className="space-y-4">
            {session.planning && (
              <div className="flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-primary">
                <Loader2 className="h-4 w-4 animate-spin" />
                正在规划研究任务...
              </div>
            )}
            {session.synthesizing && (
              <div className="flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-primary">
                <Loader2 className="h-4 w-4 animate-spin" /> 正在合成报告…
              </div>
            )}
          </div>
        </div>
      )}

      {session.isCompleted && session.finalResult && (
        <>
          <Suspense fallback={<LoadingSkeleton variant="card" count={1} />}>
            <ReportViewer result={session.finalResult} />
          </Suspense>
          {session.plan && (
            <details className="border-t border-border pt-4">
              <summary className="cursor-pointer text-sm font-semibold text-text">
                查看执行过程 · {session.plan.subtasks.length} 个子问题
              </summary>
              <div className="mt-4 space-y-4">
                <Suspense fallback={<LoadingSkeleton variant="card" count={1} />}>
                  <PlanDAG plan={session.plan} />
                </Suspense>
                <SubtaskProgressList subtasks={session.subtasks} />
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}
