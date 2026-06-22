import { cn } from "@/lib/utils";
import { CheckCircle2, Loader2, Circle, XCircle } from "lucide-react";
import type { SubTask } from "@/types/research";

interface Props {
  subtasks: Record<string, SubTask & { result?: unknown }>;
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

  if (entries.length === 0) return null;

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h4 className="mb-3 text-sm font-semibold">子任务进度</h4>
      <ul className="space-y-2">
        {entries.map((st) => {
          const Icon = statusIcon[st.status] ?? Circle;
          return (
            <li key={st.task_id} className="flex items-start gap-3 text-sm">
              <Icon
                className={cn(
                  "mt-0.5 h-4 w-4 shrink-0",
                  statusColor[st.status],
                  st.status === "in_progress" && "animate-spin",
                )}
              />
              <span
                className={cn(
                  "text-text",
                  st.status === "completed" && "text-text-muted line-through",
                )}
              >
                {st.description}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
