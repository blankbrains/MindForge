import { useState, useEffect } from "react";
import { useSettingsStore, type LLMProvider } from "@/store/settings-store";
import { Save, RotateCcw, Trash2, CheckCircle2 } from "lucide-react";

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
  useEffect(() => { loadSettings(); }, [loadSettings]);

  const handleSave = async () => {
    setSaving(true);
    const ok = await saveSettings();
    setSaving(false);
    setSaved(ok);
    if (ok) setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    const s = useSettingsStore.getState();
    s.setLLMProvider("deepseek");
    s.clearLLMApiKey();
    s.setRetrievalTopK(20);
    s.setRerankTopK(6);
    s.setMaxIterations(8);
    s.setCriticThreshold(7.0);
  };

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
            <Save className="h-4 w-4" aria-hidden="true" />{saving ? "保存中…" : saved ? "已保存 ✓" : "保存配置"}
          </button>
        </div>
      </div>

      <div className="flex gap-1 rounded-xl border border-border bg-surface-alt p-1" role="tablist" aria-label="设置分类">
        {tabs.map(({ id, label }) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={"flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors " + (tab === id ? "bg-surface text-text shadow-sm" : "text-text-muted hover:text-text")}
          >{label}</button>
        ))}
      </div>

      {tab === "llm" && <LLMTab />}
      {tab === "retrieval" && <RetrievalTab />}
      {tab === "agent" && <AgentTab />}
    </div>
  );
}

function LLMTab() {
  const provider = useSettingsStore((s) => s.llmProvider);
  const apiKey = useSettingsStore((s) => s.llmApiKey);
  const hasLLMKey = useSettingsStore((s) => s.hasLLMKey);
  const setProvider = useSettingsStore((s) => s.setLLMProvider);
  const setApiKey = useSettingsStore((s) => s.setLLMApiKey);
  const clearKey = useSettingsStore((s) => s.clearLLMApiKey);
  const [editing, setEditing] = useState(false);

  const startEdit = () => { setApiKey(""); setEditing(true); };
  const cancelEdit = () => { setApiKey(hasLLMKey ? "" : ""); setEditing(false); };

  return (
    <div className="rounded-xl border border-border bg-surface p-6 space-y-5" role="tabpanel">
      <h3 className="font-semibold">LLM 供应商配置</h3>

      <div>
        <label htmlFor="llm-provider" className="block text-sm font-medium text-text mb-1.5">供应商</label>
        <select id="llm-provider" value={provider} onChange={(e) => setProvider(e.target.value as LLMProvider)}
          className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none">
          <option value="deepseek">DeepSeek</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>

      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-sm font-medium text-text">API Key</label>
          {hasLLMKey && !editing && (
            <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/40 dark:text-green-300">
              <CheckCircle2 className="h-3 w-3" />已配置
            </span>
          )}
        </div>

        {hasLLMKey && !editing ? (
          <div className="flex items-center gap-2">
            <div className="flex-1 rounded-lg border border-green-200 bg-green-50/50 px-3 py-2 text-sm font-mono text-text-muted dark:border-green-800 dark:bg-green-950/30">
              {apiKey || "(已保存，脱敏显示)"}
            </div>
            <button type="button" onClick={startEdit}
              className="rounded-lg border border-border px-3 py-2 text-sm hover:bg-surface-alt transition-colors">
              修改
            </button>
            <button type="button" onClick={clearKey}
              className="rounded-lg border border-red-200 px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:border-red-800 dark:hover:bg-red-950 transition-colors" title="删除 Key">
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <input id="llm-api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="flex-1 rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm font-mono focus:ring-2 focus:ring-primary/20 focus:outline-none"
            />
            {editing && (
              <button type="button" onClick={cancelEdit}
                className="rounded-lg border border-border px-3 py-2 text-sm text-text-muted hover:bg-surface-alt transition-colors">
                取消
              </button>
            )}
          </div>
        )}
        <p className="mt-1 text-xs text-text-muted">
          {hasLLMKey && !editing
            ? "API Key 已保存。出于安全考虑，完整 Key 不会回显。点击「修改」可更换，点击垃圾桶可删除。"
            : "请输入 API Key。若留空则降级为文档检索模式。"}
        </p>
      </div>
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
  const threshold = useSettingsStore((s) => s.criticThreshold);
  const setMaxIter = useSettingsStore((s) => s.setMaxIterations);
  const setThreshold = useSettingsStore((s) => s.setCriticThreshold);

  return (
    <div className="rounded-xl border border-border bg-surface p-6 space-y-5" role="tabpanel">
      <h3 className="font-semibold">Agent 参数</h3>
      <div className="grid grid-cols-2 gap-5">
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
      </div>
    </div>
  );
}
