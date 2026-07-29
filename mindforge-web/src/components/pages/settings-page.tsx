import { useState } from "react";
import { LLMProviderPanel } from "@/components/settings/llm-provider-panel";
import { useSettingsStore } from "@/store/settings-store";
import {
  AlertCircle,
  Loader2,
  RotateCcw,
  Save,
} from "lucide-react";

type TabId = "llm" | "retrieval" | "agent";

const tabs: { id: TabId; label: string }[] = [
  { id: "llm", label: "LLM 供应商" },
  { id: "retrieval", label: "检索配置" },
  { id: "agent", label: "Agent 配置" },
];

export function SettingsPage() {
  const [tab, setTab] = useState<TabId>("llm");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const saveSettings = useSettingsStore((s) => s.saveSettings);
  const loadSettings = useSettingsStore((s) => s.loadSettings);
  const resetConfigDefaults = useSettingsStore(
    (s) => s.resetConfigDefaults,
  );
  const loaded = useSettingsStore((s) => s.loaded);
  const loadError = useSettingsStore((s) => s.loadError);
  const saveError = useSettingsStore((s) => s.saveError);

  const handleSave = async () => {
    setSaving(true);
    const ok = await saveSettings();
    setSaving(false);
    setSaved(ok);
    if (ok) setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    resetConfigDefaults();
  };

  if (!loaded) {
    return (
      <div
        className="flex min-h-64 items-center justify-center text-text-muted"
        role="status"
        aria-live="polite"
      >
        <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
        <span className="ml-3 text-sm">正在加载系统配置</span>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="mx-auto max-w-3xl">
        <div
          className="rounded-lg border border-red-200 bg-red-50 p-6 dark:border-red-800 dark:bg-red-950"
          role="alert"
        >
          <div className="flex items-start gap-3">
            <AlertCircle
              className="mt-0.5 h-5 w-5 shrink-0 text-red-600"
              aria-hidden="true"
            />
            <div>
              <h1 className="font-semibold text-red-800 dark:text-red-200">
                系统配置加载失败
              </h1>
              <p className="mt-1 text-sm text-red-700 dark:text-red-300">
                {loadError}
              </p>
              <button
                type="button"
                onClick={() => void loadSettings()}
                className="mt-4 inline-flex items-center gap-2 rounded-md border border-red-300 px-3 py-2 text-sm font-medium text-red-700 hover:bg-red-100 dark:border-red-700 dark:text-red-200 dark:hover:bg-red-900"
              >
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
                重新加载
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">系统配置</h1>
          <p className="mt-1 text-text-muted">管理 LLM、检索与 Agent 参数</p>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={handleReset} className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-2 text-sm font-medium text-text-muted hover:bg-surface-alt transition-colors">
            <RotateCcw className="h-4 w-4" aria-hidden="true" />重置
          </button>
          <button type="button" onClick={handleSave} disabled={saving} className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark transition-colors disabled:opacity-50">
            <Save className="h-4 w-4" aria-hidden="true" />{saving ? "保存中..." : saved ? "已保存" : "保存配置"}
          </button>
        </div>
      </div>

      {saveError && (
        <div
          className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
          role="alert"
        >
          {saveError}
        </div>
      )}

      <div className="flex gap-1 rounded-xl border border-border bg-surface-alt p-1" role="tablist" aria-label="设置分类">
        {tabs.map(({ id, label }) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={"flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors " + (tab === id ? "bg-surface text-text shadow-sm" : "text-text-muted hover:text-text")}
          >{label}</button>
        ))}
      </div>

      {tab === "llm" && <LLMProviderPanel />}
      {tab === "retrieval" && <RetrievalTab />}
      {tab === "agent" && <AgentTab />}
    </div>
  );
}

function RetrievalTab() {
  const topK = useSettingsStore((s) => s.retrievalTopK);
  const rerankK = useSettingsStore((s) => s.rerankTopK);
  const setTopK = useSettingsStore((s) => s.setRetrievalTopK);
  const setRerankK = useSettingsStore((s) => s.setRerankTopK);

  return (
    <div className="rounded-xl border border-border bg-surface p-6 space-y-5" role="tabpanel">
      <h3 className="font-semibold">检索参数</h3>
      <div className="grid grid-cols-2 gap-5">
        <div>
          <label htmlFor="retrieval-topk" className="block text-sm font-medium text-text mb-1.5">向量检索 Top-K</label>
          <input id="retrieval-topk" type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={1} max={100} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">初始检索返回的最相关文档数</p>
        </div>
        <div>
          <label htmlFor="rerank-topk" className="block text-sm font-medium text-text mb-1.5">重排序 Top-K</label>
          <input id="rerank-topk" type="number" value={rerankK} onChange={(e) => setRerankK(Number(e.target.value))} min={1} max={50} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">重排序后保留的最优文档数</p>
        </div>
      </div>
    </div>
  );
}

function AgentTab() {
  const maxIter = useSettingsStore((s) => s.maxIterations);
  const maxRefineRounds = useSettingsStore((s) => s.maxRefineRounds);
  const threshold = useSettingsStore((s) => s.criticThreshold);
  const subtaskTimeout = useSettingsStore((s) => s.subtaskTimeout);
  const researchTimeout = useSettingsStore((s) => s.researchTimeout);
  const setMaxIter = useSettingsStore((s) => s.setMaxIterations);
  const setMaxRefineRounds = useSettingsStore((s) => s.setMaxRefineRounds);
  const setThreshold = useSettingsStore((s) => s.setCriticThreshold);
  const setSubtaskTimeout = useSettingsStore((s) => s.setSubtaskTimeout);
  const setResearchTimeout = useSettingsStore((s) => s.setResearchTimeout);

  return (
    <div className="rounded-xl border border-border bg-surface p-6 space-y-5" role="tabpanel">
      <h3 className="font-semibold">Agent 参数</h3>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <div>
          <label htmlFor="agent-max-iter" className="block text-sm font-medium text-text mb-1.5">最大迭代次数</label>
          <input id="agent-max-iter" type="number" value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} min={1} max={20} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">Researcher Agent 单次任务的工具调用上限</p>
        </div>
        <div>
          <label htmlFor="agent-threshold" className="block text-sm font-medium text-text mb-1.5">评判阈值</label>
          <input id="agent-threshold" type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} min={0} max={10} step={0.1} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">Critic 评分低于此值将触发报告精炼</p>
        </div>
        <div>
          <label htmlFor="agent-refine-rounds" className="block text-sm font-medium text-text mb-1.5">最大精炼轮次</label>
          <input id="agent-refine-rounds" type="number" value={maxRefineRounds} onChange={(e) => setMaxRefineRounds(Number(e.target.value))} min={0} max={5} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">Critic 触发报告重写的最大轮数</p>
        </div>
        <div>
          <label htmlFor="agent-subtask-timeout" className="block text-sm font-medium text-text mb-1.5">子任务超时（秒）</label>
          <input id="agent-subtask-timeout" type="number" value={subtaskTimeout} onChange={(e) => setSubtaskTimeout(Number(e.target.value))} min={10} max={600} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">单个研究子任务允许执行的最长时间</p>
        </div>
        <div>
          <label htmlFor="agent-research-timeout" className="block text-sm font-medium text-text mb-1.5">研究总超时（秒）</label>
          <input id="agent-research-timeout" type="number" value={researchTimeout} onChange={(e) => setResearchTimeout(Number(e.target.value))} min={30} max={3600} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none" />
          <p className="mt-1 text-xs text-text-muted">规划、研究、综合和精炼流程的总时间上限</p>
        </div>
      </div>
    </div>
  );
}
