import { useEffect, useState } from "react";
import { useHistoryStore } from "@/store/history-store";
import { EmptyState } from "@/components/shared/empty-state";
import { Clock, Trash2, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

export function HistoryPage() {
  const { entries, loadHistory, clearAll } = useHistoryStore();
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  if (entries.length === 0) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <div><h1 className="text-3xl font-bold tracking-tight">研究历史</h1><p className="mt-1 text-text-muted">浏览过去的研究任务与结果</p></div>
        <EmptyState icon={<Clock className="h-12 w-12" />} title="暂无记录" description="完成一个研究任务后，记录会自动出现在这里" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <div><h1 className="text-3xl font-bold tracking-tight">研究历史</h1><p className="mt-1 text-text-muted">{entries.length} 条记录</p></div>
        {showClearConfirm ? (
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-muted">确认清空？</span>
            <button type="button" onClick={() => { clearAll(); setShowClearConfirm(false); }} className="rounded-lg bg-red-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-600 transition-colors">确认</button>
            <button type="button" onClick={() => setShowClearConfirm(false)} className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-muted hover:bg-surface-alt transition-colors">取消</button>
          </div>
        ) : (
          <button type="button" onClick={() => setShowClearConfirm(true)} className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-muted hover:text-red-600 hover:border-red-200 transition-colors"><Trash2 className="h-4 w-4" />清空</button>
        )}
      </div>
      <div className="space-y-3">
        {entries.map((entry) => (
          <div key={entry.id} className="rounded-xl border border-border bg-surface transition-shadow hover:shadow-sm">
            <button type="button" onClick={() => setExpandedId(expandedId === entry.id ? null : entry.id)} className="flex w-full items-center gap-4 px-5 py-4 text-left group">
              <div className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-full", entry.quality_score != null && entry.quality_score >= 7 ? "bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-400" : "bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400")}>
                <CheckCircle2 className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-medium truncate">{entry.task}</h4>
                <div className="mt-0.5 flex flex-wrap items-center gap-3 text-xs text-text-muted">
                  {entry.created_at && <span>{new Date(entry.created_at).toLocaleString()}</span>}
                  {entry.quality_score != null && <span className={cn("rounded-full px-2 py-0.5 font-medium", entry.quality_score >= 7 ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300" : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300")}>质量 {entry.quality_score.toFixed(1)}</span>}
                  {entry.model_used && <span>模型: {entry.model_used}</span>}
                </div>
              </div>
              {expandedId === entry.id ? <ChevronUp className="h-4 w-4 text-text-muted" /> : <ChevronDown className="h-4 w-4 text-text-muted" />}
            </button>
            {expandedId === entry.id && (
              <div className="border-t border-border px-5 py-4">
                <pre className="whitespace-pre-wrap text-sm font-sans bg-surface-alt rounded-lg p-4 max-h-96 overflow-y-auto">
                  {entry.report && entry.report.length > 3000 ? entry.report.slice(0, 3000) + "\n\n… (内容过长，此处截断)" : (entry.report || "（无内容）")}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
