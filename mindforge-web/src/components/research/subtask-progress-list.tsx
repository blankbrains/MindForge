import { cn } from "@/lib/utils";
import { CheckCircle2, Loader2, Circle, XCircle } from "lucide-react";
import type { SubTask } from "@/types/research";

interface Props {
  subtasks: Record<string, SubTask>;
}

const statusIcon: Record<string, React.ComponentType<{ className?: string }>> = {
  completed: CheckCircle2,
  in_progress: Loader2,
  failed: XCircle,
  pending: Circle,
};

const statusColor: Record<string, string> = {
  completed: "text-green-500",
  in_progress: "text-blue-500",
  failed: "text-red-500",
  pending: "text-border",
};

export function SubtaskProgressList({ subtasks }: Props) {
  const entries = Object.values(subtasks);
  const completed = entries.filter(
    (task) => task.status === "completed",
  ).length;

  if (entries.length === 0) return null;

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold">子问题进度</h4>
        <span className="text-xs text-text-muted">
          {completed}/{entries.length} 完成
        </span>
      </div>
      <ul className="divide-y divide-border">
        {entries.map((st, index) => {
          const Icon = statusIcon[st.status] ?? Circle;
          return (
            <li
              key={st.task_id}
              className="flex items-start gap-3 py-3 text-sm first:pt-0 last:pb-0"
            >
              <Icon
                className={cn(
                  "mt-0.5 h-4 w-4 shrink-0",
                  statusColor[st.status],
                  st.status === "in_progress" && "animate-spin",
                )}
              />
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
                  <span className="font-semibold text-text">
                    子问题 {index + 1}
                  </span>
                  <span className="font-mono">{st.task_id}</span>
                  <span>{st.task_type}</span>
                  {(st.dependencies ?? []).length > 0 && (
                    <span>依赖 {st.dependencies.join("、")}</span>
                  )}
                </div>
                <span
                  className={cn(
                    "text-text",
                    st.status === "completed"
                      && "text-text-muted line-through",
                  )}
                >
                  {st.description}
                </span>
                {st.status === "failed" && st.result?.output && (
                  <p className="mt-1 break-words text-xs text-red-600 dark:text-red-400">
                    {st.result.output}
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
