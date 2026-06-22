import { createLazyRoute } from "@tanstack/react-router";
import { EmptyState } from "@/components/shared/empty-state";
import { Clock } from "lucide-react";

function HistoryPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">研究历史</h1>
        <p className="mt-1 text-text-muted">浏览过去的研究任务与结果</p>
      </div>

      <EmptyState
        icon={<Clock className="h-12 w-12" />}
        title="暂无记录"
        description="完成一个研究任务后，记录会出现在这里"
      />
    </div>
  );
}

export const Route = createLazyRoute("/history")({
  component: HistoryPage,
});
