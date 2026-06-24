import { useCallback, useRef } from "react";
import { useResearchSession } from "@/hooks/use-research-session";
import { useResearchStore } from "@/store/research-store";
import { useSettingsStore } from "@/store/settings-store";
import { Link } from "@tanstack/react-router";
import { QueryInput } from "@/components/research/query-input";
import { PlanDAG } from "@/components/research/plan-dag";
import { SubtaskProgressList } from "@/components/research/subtask-progress-list";
import { CriticFeedbackPanel } from "@/components/research/critic-feedback-panel";
import { ReportViewer } from "@/components/research/report-viewer";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { EmptyState } from "@/components/shared/empty-state";
import { Search, Loader2, XCircle, AlertTriangle, KeyRound, FileSearch } from "lucide-react";

export function ResearchPage() {
  const session = useResearchSession();
  const task = useResearchStore((s) => s.task);
  const setTask = useResearchStore((s) => s.setTask);
  const hasLLMKey = useSettingsStore((s) => s.hasLLMKey);
  const lastTaskRef = useRef(task);

  const handleSubmit = useCallback(
    (t: string) => {
      lastTaskRef.current = t;
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

      {/* 仅在未配置 API Key 时显示警告 */}
      {!hasLLMKey && (
        <div className="flex items-center gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>未配置 LLM API Key，将使用<strong>文档检索模式</strong>。如需 AI 分析，请先</span>
          <Link to="/settings" search={{}} className="font-medium underline whitespace-nowrap">配置 API Key</Link>
        </div>
      )}

      <QueryInput value={task} onChange={setTask} onSubmit={handleSubmit} disabled={session.isStreaming} />

      {session.isError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          <XCircle className="h-5 w-5 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <p>{session.error ?? "未知错误"}</p>
            <div className="flex gap-2">
              <Link to="/settings" search={{}} className="inline-flex items-center gap-1 rounded-md bg-red-100 px-3 py-1.5 text-xs font-medium hover:bg-red-200 dark:bg-red-900 dark:hover:bg-red-800">
                <KeyRound className="h-3 w-3" />更新 API Key
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
            {session.streamingAnswer && (
              <div className="rounded-xl border border-border bg-surface p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                    {session.streamingAnswer}
                  </ReactMarkdown>
                </div>
              </div>
            )}
            <div><h4 className="mb-2 text-sm font-semibold">任务 DAG</h4><PlanDAG plan={session.plan} /></div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <SubtaskProgressList subtasks={session.subtasks} />
              <CriticFeedbackPanel score={session.criticScore} />
            </div>
            {session.refineRound > 0 && (
              <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300">
                <Loader2 className="h-4 w-4 animate-spin" /> 精炼中… 第 {session.refineRound} 轮
              </div>
            )}
          </div>
          <div className="space-y-4">
            {session.synthesizing && (
              <div className="flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-primary">
                <Loader2 className="h-4 w-4 animate-spin" /> 正在合成报告…
              </div>
            )}
          </div>
        </div>
      )}

      {session.isCompleted && session.finalResult && (
        <ReportViewer result={session.finalResult} />
      )}
    </div>
  );
}
