import { useEffect, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useHistoryStore } from "@/store/history-store";
import { EmptyState } from "@/components/shared/empty-state";
import { Modal } from "@/components/shared/modal";
import { Tooltip } from "@/components/shared/tooltip";
import { StreamingMarkdown } from "@/components/research/streaming-markdown";
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Loader2,
  Trash2,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function HistoryPage() {
  const {
    entries,
    loaded,
    loading,
    loadError,
    page,
    pageSize,
    total,
    serverTotal,
    loadHistory,
    loadEntry,
    removeEntry,
    clearAll,
  } = useHistoryStore();
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [detailLoadingId, setDetailLoadingId] = useState<number | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null);
  const [actionPending, setActionPending] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const detailRequestGenerationRef = useRef(0);

  useEffect(() => {
    void loadHistory(1);
  }, [loadHistory]);

  const toggleEntry = async (id: number) => {
    const generation = detailRequestGenerationRef.current + 1;
    detailRequestGenerationRef.current = generation;
    if (expandedId === id) {
      setExpandedId(null);
      setDetailLoadingId(null);
      return;
    }
    setActionError(null);
    setDetailLoadingId(id);
    try {
      await loadEntry(id);
      if (detailRequestGenerationRef.current !== generation) return;
      setExpandedId(id);
    } catch (error) {
      if (detailRequestGenerationRef.current !== generation) return;
      setActionError(
        error instanceof Error ? error.message : "历史详情加载失败",
      );
    } finally {
      if (detailRequestGenerationRef.current === generation) {
        setDetailLoadingId(null);
      }
    }
  };

  const confirmDelete = async () => {
    if (deleteTargetId == null) return;
    setActionPending(true);
    setActionError(null);
    try {
      await removeEntry(deleteTargetId);
      detailRequestGenerationRef.current += 1;
      setDeleteTargetId(null);
      if (expandedId === deleteTargetId) setExpandedId(null);
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : "删除历史记录失败",
      );
    } finally {
      setActionPending(false);
    }
  };

  const confirmClear = async () => {
    setActionPending(true);
    setActionError(null);
    try {
      await clearAll();
      detailRequestGenerationRef.current += 1;
      setShowClearConfirm(false);
      setExpandedId(null);
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : "清空历史记录失败",
      );
    } finally {
      setActionPending(false);
    }
  };

  if (!loaded) {
    return (
      <div className="mx-auto max-w-5xl py-16 text-center text-text-muted">
        <Loader2 className="mx-auto h-6 w-6 animate-spin" />
        <p className="mt-3 text-sm">正在加载研究历史</p>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">研究历史</h1>
          <p className="mt-1 text-text-muted">浏览过去的研究任务与结果</p>
        </div>
        {loadError ? (
          <div
            role="alert"
            className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
          >
            <p>历史记录加载失败，当前内容可能不是服务器最新状态。</p>
            <button
              type="button"
              onClick={() => void loadHistory(page)}
              disabled={loading}
              className="mt-3 rounded-lg border border-red-300 px-3 py-1.5 font-medium transition-colors hover:bg-red-100 disabled:opacity-50 dark:border-red-700 dark:hover:bg-red-900"
            >
              {loading ? "重试中…" : "重新加载"}
            </button>
          </div>
        ) : (
          <EmptyState
            icon={<Clock className="h-12 w-12" />}
            title="暂无记录"
            description="完成一个研究任务后，记录会自动出现在这里"
          />
        )}
      </div>
    );
  }

  const totalPages = Math.max(1, Math.ceil(serverTotal / pageSize));

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">研究历史</h1>
          <p className="mt-1 text-text-muted">{total} 条记录</p>
        </div>
        <button
          type="button"
          onClick={() => {
            setActionError(null);
            setShowClearConfirm(true);
          }}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-muted transition-colors hover:border-red-200 hover:text-red-600"
        >
          <Trash2 className="h-4 w-4" aria-hidden="true" />
          清空
        </button>
      </div>

      {actionError && (
        <div
          role="alert"
          className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
        >
          {actionError}
        </div>
      )}
      {loadError && (
        <div
          role="alert"
          className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
        >
          <span>服务器历史加载失败，当前显示的是上次成功结果或本地记录。</span>
          <button
            type="button"
            onClick={() => void loadHistory(page)}
            disabled={loading}
            className="rounded-lg border border-red-300 px-3 py-1.5 font-medium transition-colors hover:bg-red-100 disabled:opacity-50 dark:border-red-700 dark:hover:bg-red-900"
          >
            {loading ? "重试中…" : "重新加载"}
          </button>
        </div>
      )}

      <div className="space-y-3">
        {entries.map((entry) => {
          const expanded = expandedId === entry.id;
          const detailId = `history-detail-${entry.id}`;
          return (
            <article
              key={entry.id}
              className="group rounded-lg border border-border bg-surface transition-shadow hover:shadow-sm"
            >
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  type="button"
                  onClick={() => void toggleEntry(entry.id)}
                  aria-expanded={expanded}
                  aria-controls={detailId}
                  className="flex min-w-0 flex-1 items-center gap-4 rounded-md px-2 py-2 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                >
                  <div
                    className={cn(
                      "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                      entry.quality_score != null && entry.quality_score >= 7
                        ? "bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-400"
                        : "bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400",
                    )}
                  >
                    <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="truncate font-medium">{entry.task}</h2>
                    <div className="mt-0.5 flex flex-wrap items-center gap-3 text-xs text-text-muted">
                      {entry.created_at && (
                        <span>
                          {new Date(entry.created_at).toLocaleString()}
                        </span>
                      )}
                      {entry.quality_score != null &&
                        entry.quality_score > 0 && (
                          <Tooltip
                            content="Critic 对最终报告完整性、证据和表达质量的综合评分，不代表内容绝对正确。"
                            side="top"
                          >
                            <span
                              tabIndex={0}
                              className={cn(
                                "cursor-help rounded-full px-2 py-0.5 font-medium outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
                                entry.quality_score >= 7
                                  ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"
                                  : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
                              )}
                            >
                              质量 {entry.quality_score.toFixed(1)}
                            </span>
                          </Tooltip>
                        )}
                      {entry.quality_score == null && (
                        <Tooltip
                          content="该请求没有运行 Critic，例如会话型问题、快速模式或明确跳过质量评审的流程。"
                          side="top"
                        >
                          <span
                            tabIndex={0}
                            className="cursor-help rounded-full bg-surface-alt px-2 py-0.5 font-medium text-text-muted outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                          >
                            未评审
                          </span>
                        </Tooltip>
                      )}
                      {entry.quality_score === 0 && (
                        <Tooltip
                          content="旧版记录没有保存明确的评审状态，无法区分真实 0 分与未执行评审。"
                          side="top"
                        >
                          <span
                            tabIndex={0}
                            className="cursor-help rounded-full bg-surface-alt px-2 py-0.5 font-medium text-text-muted outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                          >
                            评分状态未知
                          </span>
                        </Tooltip>
                      )}
                      {entry.model_used && (
                        <span>模型：{entry.model_used}</span>
                      )}
                    </div>
                  </div>
                  {detailLoadingId === entry.id ? (
                    <Loader2
                      className="h-4 w-4 shrink-0 animate-spin text-text-muted"
                      aria-label="加载详情"
                    />
                  ) : expanded ? (
                    <ChevronUp
                      className="h-4 w-4 shrink-0 text-text-muted"
                      aria-hidden="true"
                    />
                  ) : (
                    <ChevronDown
                      className="h-4 w-4 shrink-0 text-text-muted"
                      aria-hidden="true"
                    />
                  )}
                </button>
                <Tooltip
                  content="永久删除这条研究历史，不会删除知识库文档。"
                  side="left"
                >
                  <button
                    type="button"
                    onClick={() => {
                      setActionError(null);
                      setDeleteTargetId(entry.id);
                    }}
                    className="shrink-0 rounded-lg p-2 text-text-muted opacity-100 transition-colors hover:bg-red-50 hover:text-red-500 focus:opacity-100 sm:opacity-0 sm:group-hover:opacity-100 dark:hover:bg-red-950"
                    aria-label={`删除研究记录：${entry.task}`}
                  >
                    <Trash2 className="h-4 w-4" aria-hidden="true" />
                  </button>
                </Tooltip>
              </div>

              {expanded && (
                <div id={detailId} className="border-t border-border px-5 py-4">
                  {entry.trace_id && (
                    <div className="mb-3 flex justify-end">
                      <Link
                        to="/observability"
                        search={{ traceId: entry.trace_id }}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-primary hover:underline"
                      >
                        <Activity className="h-3.5 w-3.5" />
                        查看 Trace
                      </Link>
                    </div>
                  )}
                  <div className="max-h-96 overflow-y-auto rounded-lg bg-surface-alt p-4 text-sm">
                    <StreamingMarkdown
                      content={entry.report || "（无内容）"}
                      sources={entry.sources}
                    />
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>

      {totalPages > 1 && (
        <nav
          aria-label="历史记录分页"
          className="flex items-center justify-between gap-4 border-t border-border pt-4"
        >
          <button
            type="button"
            onClick={() => void loadHistory(page - 1)}
            disabled={page <= 1 || loading}
            className="rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-muted transition-colors hover:bg-surface-alt disabled:opacity-50"
          >
            上一页
          </button>
          <span className="text-sm text-text-muted">
            第 {page} / {totalPages} 页
          </span>
          <button
            type="button"
            onClick={() => void loadHistory(page + 1)}
            disabled={page >= totalPages || loading}
            className="rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-muted transition-colors hover:bg-surface-alt disabled:opacity-50"
          >
            下一页
          </button>
        </nav>
      )}

      {deleteTargetId != null && (
        <Modal
          titleId="delete-history-title"
          descriptionId="delete-history-description"
          onClose={() => {
            if (!actionPending) setDeleteTargetId(null);
          }}
          closeOnBackdrop={!actionPending}
        >
          <h2 id="delete-history-title" className="text-lg font-semibold">
            确认删除
          </h2>
          <p
            id="delete-history-description"
            className="mb-4 mt-2 text-sm text-text-muted"
          >
            该研究记录将被永久删除。
          </p>
          {actionError && (
            <p role="alert" className="mb-4 text-sm text-red-600">
              {actionError}
            </p>
          )}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setDeleteTargetId(null)}
              disabled={actionPending}
              className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="button"
              onClick={() => void confirmDelete()}
              disabled={actionPending}
              className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50"
            >
              {actionPending ? "删除中…" : "确认删除"}
            </button>
          </div>
        </Modal>
      )}

      {showClearConfirm && (
        <Modal
          titleId="clear-history-title"
          descriptionId="clear-history-description"
          onClose={() => {
            if (!actionPending) setShowClearConfirm(false);
          }}
          closeOnBackdrop={!actionPending}
        >
          <h2 id="clear-history-title" className="text-lg font-semibold">
            清空研究历史
          </h2>
          <p
            id="clear-history-description"
            className="mb-4 mt-2 text-sm text-text-muted"
          >
            所有研究记录都将被永久删除，此操作无法撤销。
          </p>
          {actionError && (
            <p role="alert" className="mb-4 text-sm text-red-600">
              {actionError}
            </p>
          )}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setShowClearConfirm(false)}
              disabled={actionPending}
              className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="button"
              onClick={() => void confirmClear()}
              disabled={actionPending}
              className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50"
            >
              {actionPending ? "清空中…" : "确认清空"}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
