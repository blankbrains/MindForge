import { useDeferredValue, useMemo, useState } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  Activity,
  Bot,
  Check,
  CheckCircle2,
  Clock3,
  Copy,
  ExternalLink,
  Eye,
  EyeOff,
  Loader2,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Trash2,
  Wrench,
  XCircle,
} from "lucide-react";
import {
  useClearTraces,
  useDeleteTrace,
  useObservabilityStatus,
  useTraceDetail,
  useTraceList,
} from "@/hooks/use-observability";
import { Modal } from "@/components/shared/modal";
import { Tooltip } from "@/components/shared/tooltip";
import { cn, formatCostEstimate, formatTokenCount } from "@/lib/utils";
import type {
  TraceFailure,
  TraceObservation,
  TraceStatus,
  TraceSummary,
} from "@/types/observability";

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(2)} s`;
  return `${(durationMs / 60_000).toFixed(1)} min`;
}

function formatDate(timestamp: number): string {
  if (!Number.isFinite(timestamp)) return "未知时间";
  return new Date(timestamp * 1000).toLocaleString();
}

function statusStyle(status: TraceStatus): string {
  if (status === "success") {
    return "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300";
  }
  if (status === "warning") {
    return "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-200";
  }
  if (status === "degraded") {
    return "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200";
  }
  if (status === "cancelled") {
    return "bg-surface-alt text-text-muted";
  }
  return "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300";
}

function statusLabel(status: TraceStatus): string {
  if (status === "success") return "成功";
  if (status === "warning") return "成功（有告警）";
  if (status === "degraded") return "降级";
  if (status === "cancelled") return "已取消";
  return "失败";
}

function effectiveTraceStatus(trace: TraceSummary): TraceStatus {
  if (trace.status === "success" && trace.failure_count > 0) {
    return "warning";
  }
  return trace.status;
}

function observationLabel(name: string): string {
  const labels: Record<string, string> = {
    "orchestrator.research": "Orchestrator",
    "agent.planner": "Planner",
    "agent.researcher": "Researcher",
    "agent.synthesizer": "Synthesizer",
    "agent.critic": "Critic",
    "llm.chat": "LLM 调用",
    "tool.execute": "工具调用",
  };
  return labels[name] ?? name;
}

function ObservationIcon({ name }: { name: string }) {
  if (name === "llm.chat") return <Bot className="h-4 w-4" />;
  if (name === "tool.execute") return <Wrench className="h-4 w-4" />;
  return <Activity className="h-4 w-4" />;
}

function payloadText(value: unknown): string {
  if (value === undefined || value === null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

interface ObservationRow extends TraceObservation {
  depth: number;
}

function buildObservationRows(
  observations: TraceObservation[],
): ObservationRow[] {
  const byId = new Map(observations.map((item) => [item.span_id, item]));
  const depthCache = new Map<string, number>();
  const getDepth = (
    item: TraceObservation,
    visited = new Set<string>(),
  ): number => {
    const cached = depthCache.get(item.span_id);
    if (cached !== undefined) return cached;
    if (!item.parent_id || visited.has(item.span_id)) return 0;
    const parent = byId.get(item.parent_id);
    if (!parent) return 0;
    visited.add(item.span_id);
    const depth = Math.min(8, getDepth(parent, visited) + 1);
    depthCache.set(item.span_id, depth);
    return depth;
  };
  return observations.map((item) => ({
    ...item,
    depth: getDepth(item),
  }));
}

function TraceListItem({
  trace,
  selected,
  onSelect,
}: {
  trace: TraceSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  const displayStatus = effectiveTraceStatus(trace);
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "w-full border-b border-border px-4 py-3 text-left transition-colors last:border-b-0",
        selected
          ? "bg-primary/8"
          : "hover:bg-surface-alt focus-visible:bg-surface-alt",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">
            {trace.display_name ||
              trace.task_preview ||
              observationLabel(trace.name)}
          </p>
          <p className="mt-1 truncate font-mono text-[11px] text-text-muted">
            {trace.trace_id}
          </p>
        </div>
        <span
          className={cn(
            "shrink-0 rounded px-2 py-1 text-[11px] font-semibold",
            statusStyle(displayStatus),
          )}
        >
          {statusLabel(displayStatus)}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-muted">
        <span>{formatDate(trace.start_time)}</span>
        <span>{formatDuration(trace.duration_ms)}</span>
        <span>{trace.span_count} 个观察</span>
      </div>
    </button>
  );
}

export function ObservabilityPage() {
  const routeSearch = useSearch({ from: "/observability" });
  const navigate = useNavigate();
  const [searchInput, setSearchInput] = useState("");
  const [statusFilter, setStatusFilter] = useState<TraceStatus | "">("");
  const [copied, setCopied] = useState(false);
  const [deleteMode, setDeleteMode] = useState<"selected" | "all" | null>(null);
  const deferredSearch = useDeferredValue(searchInput);
  const statusQuery = useObservabilityStatus();
  const listQuery = useTraceList({
    search: deferredSearch,
    status: statusFilter,
  });
  const selectedTraceId =
    routeSearch.traceId ?? listQuery.data?.traces[0]?.trace_id ?? null;
  const detailQuery = useTraceDetail(selectedTraceId);
  const deleteTrace = useDeleteTrace();
  const clearTraces = useClearTraces();

  const observations = useMemo(
    () => buildObservationRows(detailQuery.data?.observations ?? []),
    [detailQuery.data?.observations],
  );
  const failuresBySpan = useMemo(
    () =>
      new Map<string, TraceFailure>(
        (detailQuery.data?.failures ?? []).map((failure) => [
          failure.span_id,
          failure,
        ]),
      ),
    [detailQuery.data?.failures],
  );
  const traceStart = detailQuery.data?.summary.start_time ?? 0;
  const traceDuration = Math.max(detailQuery.data?.summary.duration_ms ?? 0, 1);

  const selectTrace = (traceId: string) => {
    void navigate({
      to: "/observability",
      search: { traceId },
      replace: true,
    });
  };

  const refresh = () => {
    void statusQuery.refetch();
    void listQuery.refetch();
    if (selectedTraceId) void detailQuery.refetch();
  };

  const copyTraceId = async () => {
    if (!selectedTraceId) return;
    try {
      await navigator.clipboard.writeText(selectedTraceId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const confirmDelete = async () => {
    if (deleteMode === "selected" && selectedTraceId) {
      await deleteTrace.mutateAsync(selectedTraceId);
    } else if (deleteMode === "all") {
      await clearTraces.mutateAsync();
    }
    setDeleteMode(null);
    void navigate({
      to: "/observability",
      search: {},
      replace: true,
    });
  };
  const deletePending = deleteTrace.isPending || clearTraces.isPending;

  const status = statusQuery.data;
  const summary = detailQuery.data?.summary;
  const failureSummary = summary?.failure_summary ?? summary?.error ?? null;
  const selectedTraceStatus = summary ? effectiveTraceStatus(summary) : "error";

  return (
    <div className="mx-auto max-w-[1500px] space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">可观测</h1>
          <p className="mt-1 text-text-muted">
            查看研究任务的 Orchestrator、Agent、工具与模型调用链
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tooltip
            content="删除全部本地执行链路；研究历史和 Langfuse 远程数据不受影响。"
            side="bottom"
          >
            <button
              type="button"
              onClick={() => setDeleteMode("all")}
              disabled={!listQuery.data?.traces.length}
              aria-label="清空本地 Trace"
              className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:border-red-200 hover:text-red-600 disabled:opacity-40"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </Tooltip>
          <Tooltip
            content="重新读取可观测状态、Trace 列表和当前链路详情。"
            side="bottom"
          >
            <button
              type="button"
              onClick={refresh}
              disabled={statusQuery.isFetching || listQuery.isFetching}
              aria-label="刷新追踪数据"
              className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:bg-surface-alt hover:text-text disabled:opacity-50"
            >
              <RefreshCw
                className={cn(
                  "h-4 w-4",
                  (statusQuery.isFetching || listQuery.isFetching) &&
                    "animate-spin",
                )}
              />
            </button>
          </Tooltip>
        </div>
      </header>

      {statusQuery.isError ? (
        <div
          role="alert"
          className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
        >
          无法读取可观测状态，请检查后端服务。
        </div>
      ) : (
        <section
          aria-label="可观测状态"
          className="grid grid-cols-2 border border-border bg-surface sm:grid-cols-4"
        >
          <div className="border-b border-r border-border px-4 py-3 sm:border-b-0">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <Activity className="h-4 w-4" />
              本地追踪
            </div>
            <p className="mt-1 text-sm font-semibold">
              {status?.enabled ? "已启用" : "已停用"}
            </p>
          </div>
          <div className="border-b border-border px-4 py-3 sm:border-b-0 sm:border-r">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <Server className="h-4 w-4" />
              Langfuse
            </div>
            <p className="mt-1 text-sm font-semibold">
              {status?.remote_configured ? "已配置" : "未配置"}
            </p>
          </div>
          <div className="border-r border-border px-4 py-3">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              {status?.capture_content ? (
                <Eye className="h-4 w-4" />
              ) : (
                <EyeOff className="h-4 w-4" />
              )}
              内容采集
            </div>
            <p className="mt-1 text-sm font-semibold">
              {status?.capture_content ? "已启用" : "默认隐藏"}
            </p>
          </div>
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <ShieldCheck className="h-4 w-4" />
              本地保留
            </div>
            <p className="mt-1 text-sm font-semibold">
              {status
                ? status.retention_days === 0
                  ? "永久"
                  : `${status.retention_days} 天`
                : "加载中"}
            </p>
          </div>
        </section>
      )}

      <div className="grid min-h-[620px] grid-cols-1 border border-border bg-surface lg:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="border-b border-border lg:border-b-0 lg:border-r">
          <div className="space-y-3 border-b border-border p-3">
            <label className="relative block">
              <span className="sr-only">搜索 Trace ID 或任务</span>
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-text-muted" />
              <input
                type="search"
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="搜索 Trace ID 或任务"
                className="h-9 w-full rounded-md border border-border bg-surface-alt pl-9 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/15"
              />
            </label>
            <label className="block">
              <span className="sr-only">按执行状态筛选</span>
              <select
                value={statusFilter}
                onChange={(event) =>
                  setStatusFilter(event.target.value as TraceStatus | "")
                }
                className="h-9 w-full rounded-md border border-border bg-surface px-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/15"
              >
                <option value="">全部状态</option>
                <option value="success">成功</option>
                <option value="warning">成功（有告警）</option>
                <option value="degraded">降级</option>
                <option value="error">失败</option>
                <option value="cancelled">已取消</option>
              </select>
            </label>
          </div>

          <div className="max-h-[560px] overflow-y-auto">
            {listQuery.isLoading ? (
              <div className="flex items-center justify-center gap-2 py-16 text-sm text-text-muted">
                <Loader2 className="h-4 w-4 animate-spin" />
                正在加载 Trace
              </div>
            ) : listQuery.isError ? (
              <div
                role="alert"
                className="px-5 py-12 text-center text-sm text-red-600"
              >
                Trace 列表加载失败。
              </div>
            ) : listQuery.data?.traces.length ? (
              listQuery.data.traces.map((trace) => (
                <TraceListItem
                  key={trace.trace_id}
                  trace={trace}
                  selected={selectedTraceId === trace.trace_id}
                  onSelect={() => selectTrace(trace.trace_id)}
                />
              ))
            ) : (
              <div className="px-5 py-16 text-center">
                <Activity className="mx-auto h-8 w-8 text-text-muted" />
                <p className="mt-3 text-sm font-medium">暂无匹配的 Trace</p>
                <p className="mt-1 text-xs text-text-muted">
                  完成一次研究后，链路会显示在这里。
                </p>
              </div>
            )}
          </div>
          {listQuery.data?.truncated && (
            <p className="border-t border-border px-4 py-2 text-xs text-text-muted">
              列表已按后端扫描上限截断，请使用搜索缩小范围。
            </p>
          )}
        </aside>

        <main className="min-w-0">
          {!selectedTraceId ? (
            <div className="flex min-h-[560px] flex-col items-center justify-center px-6 text-center">
              <Activity className="h-10 w-10 text-text-muted" />
              <p className="mt-3 font-medium">选择一个 Trace 查看完整链路</p>
            </div>
          ) : detailQuery.isLoading ? (
            <div className="flex min-h-[560px] items-center justify-center gap-2 text-sm text-text-muted">
              <Loader2 className="h-5 w-5 animate-spin" />
              正在加载 Trace 详情
            </div>
          ) : detailQuery.isError || !summary ? (
            <div
              role="alert"
              className="flex min-h-[560px] flex-col items-center justify-center px-6 text-center text-red-600"
            >
              <XCircle className="h-9 w-9" />
              <p className="mt-3 font-medium">Trace 详情加载失败</p>
              <button
                type="button"
                onClick={() => void detailQuery.refetch()}
                className="mt-4 rounded-md border border-border px-3 py-2 text-sm text-text hover:bg-surface-alt"
              >
                重试
              </button>
            </div>
          ) : (
            <>
              <div className="border-b border-border p-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="text-lg font-semibold">
                        {summary.display_name ||
                          summary.task_preview ||
                          observationLabel(summary.name)}
                      </h2>
                      <span
                        className={cn(
                          "rounded px-2 py-1 text-xs font-semibold",
                          statusStyle(selectedTraceStatus),
                        )}
                      >
                        {statusLabel(selectedTraceStatus)}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <code className="break-all text-xs text-text-muted">
                        {summary.trace_id}
                      </code>
                      <Tooltip
                        content={
                          copied ? "Trace ID 已复制" : "复制完整 Trace ID"
                        }
                        side="top"
                      >
                        <button
                          type="button"
                          onClick={() => void copyTraceId()}
                          aria-label="复制 Trace ID"
                          className="rounded p-1 text-text-muted hover:bg-surface-alt hover:text-text"
                        >
                          {copied ? (
                            <Check className="h-4 w-4 text-emerald-600" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </button>
                      </Tooltip>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {summary.remote_url && (
                      <a
                        href={summary.remote_url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-alt"
                      >
                        在 Langfuse 中打开
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                    <Tooltip
                      content="只删除当前本地执行链路；研究历史和 Langfuse 远程数据不受影响。"
                      side="left"
                    >
                      <button
                        type="button"
                        onClick={() => setDeleteMode("selected")}
                        aria-label="删除当前 Trace"
                        className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-border text-text-muted hover:border-red-200 hover:text-red-600"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </Tooltip>
                  </div>
                </div>

                {failureSummary && (
                  <div
                    className={cn(
                      "mt-4 border px-4 py-3 text-sm",
                      selectedTraceStatus === "degraded" ||
                        selectedTraceStatus === "warning" ||
                        selectedTraceStatus === "success"
                        ? "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200"
                        : "border-red-200 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300",
                    )}
                  >
                    <span className="font-semibold">
                      {selectedTraceStatus === "degraded"
                        ? "降级原因汇总："
                        : selectedTraceStatus === "warning" ||
                            selectedTraceStatus === "success"
                          ? "链路异常汇总（最终已恢复）："
                          : "失败原因汇总："}
                    </span>
                    {failureSummary}
                    {summary.failure_count > 0 && (
                      <p className="mt-1 text-xs opacity-80">
                        展开下方红色执行节点可查看阶段、错误类型、模型和重试次数。
                      </p>
                    )}
                  </div>
                )}

                <dl className="mt-5 grid grid-cols-2 gap-px overflow-hidden border border-border bg-border sm:grid-cols-4 xl:grid-cols-7">
                  {[
                    ["开始时间", formatDate(summary.start_time)],
                    ["总耗时", formatDuration(summary.duration_ms)],
                    ["观察数", String(summary.span_count)],
                    ["模型调用", String(summary.generation_count)],
                    ["工具调用", String(summary.tool_count)],
                    ["Token", formatTokenCount(summary.total_tokens)],
                    [
                      "费用",
                      formatCostEstimate(summary.cost_usd, summary.cost_status),
                    ],
                  ].map(([label, value]) => (
                    <div key={label} className="bg-surface px-3 py-3">
                      <dt className="text-xs text-text-muted">{label}</dt>
                      <dd className="mt-1 text-sm font-semibold">{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <section aria-labelledby="trace-chain-title">
                <div className="flex items-center justify-between border-b border-border px-5 py-3">
                  <h3 id="trace-chain-title" className="text-sm font-semibold">
                    执行链路
                  </h3>
                  {detailQuery.data?.observations_truncated && (
                    <span className="text-xs text-amber-600">
                      部分观察已按上限截断
                    </span>
                  )}
                </div>
                <div className="max-h-[560px] overflow-auto">
                  {observations.length === 0 ? (
                    <p className="px-5 py-12 text-center text-sm text-text-muted">
                      该 Trace 暂无观察记录。
                    </p>
                  ) : (
                    observations.map((observation) => {
                      const failure = failuresBySpan.get(observation.span_id);
                      const isFailure = Boolean(failure || observation.error);
                      const offset = Math.max(
                        0,
                        (((observation.start_time - traceStart) * 1000) /
                          traceDuration) *
                          100,
                      );
                      const width = Math.max(
                        1,
                        Math.min(
                          100 - offset,
                          (observation.duration_ms / traceDuration) * 100,
                        ),
                      );
                      const hasPayload = Boolean(
                        isFailure ||
                        observation.input !== undefined ||
                        observation.output !== undefined ||
                        Object.keys(observation.metadata).length,
                      );
                      return (
                        <details
                          key={observation.span_id}
                          className="group border-b border-border last:border-b-0"
                        >
                          <summary
                            className={cn(
                              "grid min-w-[760px] cursor-pointer grid-cols-[minmax(220px,1fr)_90px_minmax(220px,1fr)_30px] items-center gap-3 px-5 py-3 text-sm hover:bg-surface-alt",
                              !hasPayload && "cursor-default",
                            )}
                            onClick={(event) => {
                              if (!hasPayload) event.preventDefault();
                            }}
                          >
                            <span
                              className="flex min-w-0 items-center gap-2"
                              style={{
                                paddingLeft: `${observation.depth * 18}px`,
                              }}
                            >
                              <span
                                className={cn(
                                  "shrink-0",
                                  isFailure ? "text-red-600" : "text-primary",
                                )}
                              >
                                <ObservationIcon name={observation.name} />
                              </span>
                              <span className="truncate font-medium">
                                {observationLabel(observation.name)}
                              </span>
                            </span>
                            <span className="text-right font-mono text-xs text-text-muted">
                              {formatDuration(observation.duration_ms)}
                            </span>
                            <span className="relative h-5 overflow-hidden bg-surface-alt">
                              <span
                                className={cn(
                                  "absolute top-1 h-3",
                                  isFailure
                                    ? "bg-red-400"
                                    : observation.name === "llm.chat"
                                      ? "bg-cyan-500"
                                      : observation.name === "tool.execute"
                                        ? "bg-amber-500"
                                        : "bg-primary",
                                )}
                                style={{
                                  left: `${Math.min(offset, 99)}%`,
                                  width: `${width}%`,
                                }}
                              />
                            </span>
                            <span className="flex justify-end">
                              {isFailure ? (
                                <XCircle className="h-4 w-4 text-red-600" />
                              ) : (
                                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                              )}
                            </span>
                          </summary>
                          {hasPayload && (
                            <div className="border-t border-border bg-surface-alt px-5 py-4">
                              {failure && (
                                <div
                                  role="alert"
                                  className="mb-4 border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
                                >
                                  <p className="font-semibold">
                                    {failure.message}
                                  </p>
                                  <dl className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs">
                                    <div>
                                      <dt className="inline opacity-70">
                                        阶段：
                                      </dt>
                                      <dd className="inline">
                                        {failure.stage}
                                      </dd>
                                    </div>
                                    <div>
                                      <dt className="inline opacity-70">
                                        错误码：
                                      </dt>
                                      <dd className="inline font-mono">
                                        {failure.error_code}
                                      </dd>
                                    </div>
                                    <div>
                                      <dt className="inline opacity-70">
                                        类型：
                                      </dt>
                                      <dd className="inline font-mono">
                                        {failure.error_type}
                                      </dd>
                                    </div>
                                    {failure.agent && (
                                      <div>
                                        <dt className="inline opacity-70">
                                          Agent：
                                        </dt>
                                        <dd className="inline">
                                          {failure.agent}
                                        </dd>
                                      </div>
                                    )}
                                    {failure.model && (
                                      <div>
                                        <dt className="inline opacity-70">
                                          模型：
                                        </dt>
                                        <dd className="inline">
                                          {failure.model}
                                        </dd>
                                      </div>
                                    )}
                                    {failure.attempt && (
                                      <div>
                                        <dt className="inline opacity-70">
                                          尝试：
                                        </dt>
                                        <dd className="inline">
                                          第 {failure.attempt} 次
                                        </dd>
                                      </div>
                                    )}
                                  </dl>
                                </div>
                              )}
                              <div className="grid gap-4 xl:grid-cols-3">
                                {[
                                  ["输入", observation.input],
                                  ["输出", observation.output],
                                  ["元数据", observation.metadata],
                                ].map(([label, value]) => {
                                  const text = payloadText(value);
                                  if (!text || text === "{}") return null;
                                  return (
                                    <div
                                      key={String(label)}
                                      className="min-w-0"
                                    >
                                      <p className="mb-2 text-xs font-semibold text-text-muted">
                                        {String(label)}
                                      </p>
                                      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-md border border-border bg-surface p-3 font-mono text-xs leading-5">
                                        {text}
                                      </pre>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                        </details>
                      );
                    })
                  )}
                </div>
              </section>
            </>
          )}
        </main>
      </div>

      <div className="flex items-center gap-2 text-xs text-text-muted">
        <Clock3 className="h-4 w-4" />
        页面自动刷新；原始研究内容是否可见由后端内容采集开关决定。
      </div>

      {deleteMode && (
        <Modal
          titleId="delete-trace-title"
          descriptionId="delete-trace-description"
          onClose={() => {
            if (!deletePending) setDeleteMode(null);
          }}
          closeOnBackdrop={!deletePending}
        >
          <h2 id="delete-trace-title" className="text-lg font-semibold">
            {deleteMode === "all" ? "清空本地 Trace" : "删除当前 Trace"}
          </h2>
          <p
            id="delete-trace-description"
            className="mb-5 mt-2 text-sm leading-6 text-text-muted"
          >
            {deleteMode === "all"
              ? "所有本地执行链路都将被删除，研究历史与 Langfuse 远程数据不受影响。"
              : "当前本地执行链路将被删除，研究历史与 Langfuse 远程数据不受影响。"}
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setDeleteMode(null)}
              disabled={deletePending}
              className="flex-1 rounded-md border border-border px-4 py-2.5 text-sm font-medium hover:bg-surface-alt disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="button"
              onClick={() => void confirmDelete()}
              disabled={deletePending}
              className="flex-1 rounded-md bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50"
            >
              {deletePending ? "删除中" : "确认删除"}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
