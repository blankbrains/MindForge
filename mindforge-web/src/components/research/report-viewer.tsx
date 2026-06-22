import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { AgentResult } from "@/types/research";
import { formatDuration, formatCost } from "@/lib/utils";

interface Props {
  result: AgentResult | null;
}

export function ReportViewer({ result }: Props) {
  if (!result) return null;

  const metadata =
    (result.metadata ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-4">
      {/* Metadata bar */}
      <div className="flex flex-wrap gap-4 rounded-xl border border-border bg-surface p-4 text-sm">
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
        {result.cost_usd !== undefined && (
          <div>
            <span className="text-text-muted">费用：</span>
            <span className="font-semibold">
              {formatCost(result.cost_usd)}
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
      <div className="rounded-xl border border-border bg-surface p-6 lg:p-8">
        <div className="prose prose-neutral dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
          >
            {result.output}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
