import type { AgentResult } from "@/types/research";
import { Link } from "@tanstack/react-router";
import { Activity, AlertTriangle } from "lucide-react";
import { StreamingMarkdown } from "@/components/research/streaming-markdown";
import { normalizeCitationSources } from "@/lib/citations";
import {
  formatCostEstimate,
  formatDuration,
  formatTokenCount,
} from "@/lib/utils";

interface Props {
  result: AgentResult | null;
}

export function ReportViewer({ result }: Props) {
  if (!result) return null;

  const metadata =
    (result.metadata ?? {}) as Record<string, unknown>;
  const totalTokens = result.token_usage?.total_tokens
    ?? (
      (result.token_usage?.prompt_tokens ?? 0)
      + (result.token_usage?.completion_tokens ?? 0)
    );
  const hasTokenUsage = totalTokens > 0;
  const costStatus = result.cost_status
    ?? (
      typeof metadata.cost_status === "string"
        ? metadata.cost_status
        : undefined
    );
  const sources = normalizeCitationSources(result.data?.sources);
  const outcome =
    typeof metadata.outcome === "string" ? metadata.outcome : "success";
  const degraded = outcome === "degraded";
  const failureReason =
    typeof metadata.failure_reason === "string"
      ? metadata.failure_reason
      : typeof result.data?.primary_failure === "string"
        ? result.data.primary_failure
        : "";
  const retrievalQuality =
    typeof metadata.retrieval_quality === "number"
      ? metadata.retrieval_quality
      : typeof result.data?.retrieval_quality === "number"
        ? result.data.retrieval_quality
        : null;
  const rawQuality =
    typeof metadata.quality === "number" && Number.isFinite(metadata.quality)
      ? metadata.quality
      : null;
  const criticScore = result.data?.critic_score;
  const explicitQualityStatus =
    typeof metadata.quality_status === "string"
      ? metadata.quality_status
      : null;
  const qualityStatus =
    explicitQualityStatus
    ?? (
      rawQuality !== null
      && (
        rawQuality !== 0
        || (criticScore !== null && typeof criticScore === "object")
      )
        ? "evaluated"
        : "not_evaluated"
    );
  const quality = qualityStatus === "evaluated" ? rawQuality : null;
  const fromCache = result.data?.from_cache === true;
  const completedSubtasks =
    typeof metadata.completed_subtask_count === "number"
      ? metadata.completed_subtask_count
      : null;
  const failedSubtasks =
    typeof metadata.failed_subtask_count === "number"
      ? metadata.failed_subtask_count
      : 0;
  const fallback = result.data?.fallback === true;
  const refinementFailure =
    typeof result.data?.refinement_failure === "string"
      ? result.data.refinement_failure
      : "";

  return (
    <div className="space-y-4">
      {degraded && (
        <div
          role="status"
          className="flex items-start gap-3 border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200"
        >
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          <div>
            <p className="font-semibold">
              {fallback
                ? "研究链路失败，当前展示知识库检索结果"
                : refinementFailure
                  ? "报告精炼未完成，当前展示评审前的有效版本"
                  : "部分子任务未完成，当前报告为降级结果"}
            </p>
            {failureReason && (
              <p className="mt-1 break-words text-xs opacity-80">
                {failureReason}
              </p>
            )}
          </div>
        </div>
      )}
      {/* Metadata bar */}
      <div className="flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-border bg-surface px-4 py-3 text-sm">
        {fromCache && (
          <div>
            <span className="text-text-muted">结果来源：</span>
            <span className="font-semibold">缓存命中</span>
          </div>
        )}
        <div>
          <span className="text-text-muted">报告评分：</span>
          {quality !== null ? (
            <span className="font-semibold text-primary">
              {quality.toFixed(1)} / 10
            </span>
          ) : (
            <span className="font-semibold text-text-muted">
              {qualityStatus === "evaluation_failed" ? "评审失败" : "未评审"}
            </span>
          )}
        </div>
        {retrievalQuality !== null && (
          <div>
            <span className="text-text-muted">检索相关性：</span>
            <span className="font-semibold">
              {retrievalQuality.toFixed(1)} / 10
            </span>
          </div>
        )}
        {result.latency_ms !== undefined && (
          <div>
            <span className="text-text-muted">耗时：</span>
            <span className="font-semibold">
              {formatDuration(result.latency_ms)}
            </span>
          </div>
        )}
        {(result.cost_usd != null || costStatus) && (
          <div>
            <span className="text-text-muted">估算费用：</span>
            <span className="font-semibold">
              {formatCostEstimate(result.cost_usd, costStatus)}
            </span>
          </div>
        )}
        {hasTokenUsage && (
          <div>
            <span className="text-text-muted">Token：</span>
            <span className="font-semibold">
              {formatTokenCount(totalTokens)}
            </span>
          </div>
        )}
        {metadata.subtask_count !== undefined && (
          <div>
            <span className="text-text-muted">子任务：</span>
            {completedSubtasks !== null ? (
              <span className="font-semibold">
                {completedSubtasks}/{String(metadata.subtask_count)} 完成
                {failedSubtasks > 0 ? `，${failedSubtasks} 失败` : ""}
              </span>
            ) : (
              <span className="font-semibold">
                {String(metadata.subtask_count)}
              </span>
            )}
          </div>
        )}
        {result.trace_id && (
          <Link
            to="/observability"
            search={{ traceId: result.trace_id }}
            className="ml-auto inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
          >
            <Activity className="h-4 w-4" />
            查看 Trace
          </Link>
        )}
      </div>

      {/* Report content */}
      <div className="rounded-lg border border-border bg-surface p-5 sm:p-7 lg:p-8">
        <StreamingMarkdown content={result.output} sources={sources} />
      </div>
    </div>
  );
}
