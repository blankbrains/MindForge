import type { AgentResult } from "@/types/research";
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

  return (
    <div className="space-y-4">
      {/* Metadata bar */}
      <div className="flex flex-wrap gap-x-6 gap-y-2 rounded-lg border border-border bg-surface px-4 py-3 text-sm">
        {metadata.quality !== undefined && (
          <div>
            <span className="text-text-muted">质量评分：</span>
            <span className="font-semibold text-primary">
              {Number(metadata.quality).toFixed(1)} / 10
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
            <span className="text-text-muted">任务数：</span>
            <span className="font-semibold">
              {String(metadata.subtask_count)}
            </span>
          </div>
        )}
      </div>

      {/* Report content */}
      <div className="rounded-lg border border-border bg-surface p-5 sm:p-7 lg:p-8">
        <StreamingMarkdown content={result.output} sources={sources} />
      </div>
    </div>
  );
}
