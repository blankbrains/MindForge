import type { AgentResult } from "@/types/research";
import { Link } from "@tanstack/react-router";
import { Activity, AlertTriangle, Info } from "lucide-react";
import { StreamingMarkdown } from "@/components/research/streaming-markdown";
import { Tooltip } from "@/components/shared/tooltip";
import { normalizeCitationSources } from "@/lib/citations";
import {
  formatCostEstimate,
  formatDuration,
  formatTokenCount,
} from "@/lib/utils";

interface Props {
  result: AgentResult | null;
}

function MetricLabel({
  children,
  explanation,
}: {
  children: string;
  explanation: string;
}) {
  return (
    <Tooltip content={explanation} side="top">
      <span
        tabIndex={0}
        className="cursor-help text-text-muted underline decoration-dotted underline-offset-4 outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        {children}
      </span>
    </Tooltip>
  );
}

export function ReportViewer({ result }: Props) {
  if (!result) return null;

  const metadata = (result.metadata ?? {}) as Record<string, unknown>;
  const totalTokens =
    result.token_usage?.total_tokens ??
    (result.token_usage?.prompt_tokens ?? 0) +
      (result.token_usage?.completion_tokens ?? 0);
  const hasTokenUsage = totalTokens > 0;
  const costStatus =
    result.cost_status ??
    (typeof metadata.cost_status === "string"
      ? metadata.cost_status
      : undefined);
  const sources = normalizeCitationSources(result.data?.sources);
  const outcome =
    typeof metadata.outcome === "string" ? metadata.outcome : "success";
  const degraded = outcome === "degraded";
  const groundingStatus =
    typeof metadata.grounding_status === "string"
      ? metadata.grounding_status
      : typeof result.data?.grounding_status === "string"
        ? result.data.grounding_status
        : "";
  const modelOnly = groundingStatus === "model_only";
  const sourceWarning =
    typeof metadata.source_warning === "string"
      ? metadata.source_warning
      : typeof result.data?.source_warning === "string"
        ? result.data.source_warning
        : "";
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
    explicitQualityStatus ??
    (rawQuality !== null &&
    (rawQuality !== 0 ||
      (criticScore !== null && typeof criticScore === "object"))
      ? "evaluated"
      : "not_evaluated");
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
  const route = typeof metadata.route === "string" ? metadata.route : "";
  const directAnswer = route === "direct_answer";
  const conversationalAnswer = route === "conversation";

  return (
    <div className="space-y-4">
      {modelOnly && (
        <div
          role="status"
          className="flex items-start gap-3 border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200"
        >
          {degraded ? (
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          ) : (
            <Info className="mt-0.5 h-5 w-5 shrink-0" />
          )}
          <div>
            <p className="font-semibold">
              {degraded
                ? "来源检索未完成，当前为模型知识回答"
                : "未获得可核验来源，当前为模型知识回答"}
            </p>
            <p className="mt-1 break-words text-xs opacity-80">
              {degraded && failureReason
                ? failureReason
                : "回答未引用知识库或联网来源，请勿将其视为已核验事实。"}
              {!degraded && sourceWarning ? `（${sourceWarning}）` : ""}
            </p>
          </div>
        </div>
      )}
      {degraded && !modelOnly && (
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
            <MetricLabel explanation="该结果来自完全匹配的历史缓存，没有重新执行 Agent 研究流程。">
              结果来源：
            </MetricLabel>
            <span className="font-semibold">缓存命中</span>
          </div>
        )}
        {directAnswer || conversationalAnswer ? (
          <div>
            <MetricLabel
              explanation={
                directAnswer
                  ? "该请求使用无需外部检索的轻量直答链路，保留单任务计划与 Critic 评审。"
                  : "该请求是确定性的会话响应，不进入研究、规划或评审。"
              }
            >
              回答模式：
            </MetricLabel>
            <span className="font-semibold">
              {directAnswer ? "轻量直答" : "会话响应"}
            </span>
          </div>
        ) : null}
        {!conversationalAnswer && (
          <div>
            <MetricLabel explanation="Critic 对报告完整性、证据、结构和表达的综合评分。会话型任务和部分模式不会运行评审。">
              报告评分：
            </MetricLabel>
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
        )}
        {retrievalQuality !== null && (
          <div>
            <MetricLabel explanation="检索结果与当前问题的综合相关程度，用于判断证据是否足够支持回答。">
              检索相关性：
            </MetricLabel>
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
            <MetricLabel explanation="根据模型用量和已配置价格估算，仅供参考；部分供应商或工具调用可能无法计价。">
              估算费用：
            </MetricLabel>
            <span className="font-semibold">
              {formatCostEstimate(result.cost_usd, costStatus)}
            </span>
          </div>
        )}
        {hasTokenUsage && (
          <div>
            <MetricLabel explanation="本轮模型调用消耗的输入与输出 Token 总量，不包括无法返回用量的调用。">
              Token：
            </MetricLabel>
            <span className="font-semibold">
              {formatTokenCount(totalTokens)}
            </span>
          </div>
        )}
        {!conversationalAnswer && metadata.subtask_count !== undefined && (
            <div>
              <MetricLabel explanation="Planner 将原始问题拆分出的研究任务数量，以及最终成功完成的数量。">
                子任务：
              </MetricLabel>
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
