import { useState } from "react";
import { AlertCircle, Loader2, RotateCcw, Save } from "lucide-react";
import { LLMProviderPanel } from "@/components/settings/llm-provider-panel";
import { ObservabilitySettingsPanel } from "@/components/settings/observability-settings-panel";
import { ResearchSettingsPanel } from "@/components/settings/research-settings-panel";
import { RetrievalSettingsPanel } from "@/components/settings/retrieval-settings-panel";
import { Modal } from "@/components/shared/modal";
import { useSettingsStore } from "@/store/settings-store";

type TabId = "llm" | "retrieval" | "research" | "observability";

const tabs: { id: TabId; label: string }[] = [
  { id: "llm", label: "模型" },
  { id: "retrieval", label: "检索" },
  { id: "research", label: "研究流程" },
  { id: "observability", label: "可观测" },
];

export function SettingsPage() {
  const [tab, setTab] = useState<TabId>("llm");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const saveSettings = useSettingsStore((state) => state.saveSettings);
  const loadSettings = useSettingsStore((state) => state.loadSettings);
  const resetConfigDefaults = useSettingsStore(
    (state) => state.resetConfigDefaults,
  );
  const loaded = useSettingsStore((state) => state.loaded);
  const loadError = useSettingsStore((state) => state.loadError);
  const saveError = useSettingsStore((state) => state.saveError);

  const handleSave = async () => {
    setSaving(true);
    const ok = await saveSettings();
    setSaving(false);
    setSaved(ok);
    if (ok) window.setTimeout(() => setSaved(false), 2000);
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
          className="border border-red-200 bg-red-50 p-6 dark:border-red-800 dark:bg-red-950"
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
    <div className="mx-auto max-w-4xl space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">系统配置</h1>
          <p className="mt-1 text-text-muted">
            管理模型、检索、研究流程与可观测参数
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowResetConfirm(true)}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-2 text-sm font-medium text-text-muted hover:bg-surface-alt"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            重置
          </button>
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={saving}
            className="inline-flex min-w-28 items-center justify-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark disabled:opacity-50"
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Save className="h-4 w-4" aria-hidden="true" />
            )}
            {saving ? "保存中" : saved ? "已保存" : "保存配置"}
          </button>
        </div>
      </header>

      {saveError && (
        <div
          className="border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
          role="alert"
        >
          {saveError}
        </div>
      )}

      <div
        className="grid grid-cols-2 gap-1 border border-border bg-surface-alt p-1 sm:grid-cols-4"
        role="tablist"
        aria-label="设置分类"
      >
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={
              "min-h-9 rounded-md px-3 py-2 text-sm font-medium transition-colors "
              + (
                tab === id
                  ? "bg-surface text-text shadow-sm"
                  : "text-text-muted hover:text-text"
              )
            }
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "llm" && <LLMProviderPanel />}
      {tab === "retrieval" && <RetrievalSettingsPanel />}
      {tab === "research" && <ResearchSettingsPanel />}
      {tab === "observability" && <ObservabilitySettingsPanel />}

      {showResetConfirm && (
        <Modal
          titleId="reset-settings-title"
          descriptionId="reset-settings-description"
          onClose={() => setShowResetConfirm(false)}
        >
          <h2 id="reset-settings-title" className="text-lg font-semibold">
            重置可调参数
          </h2>
          <p
            id="reset-settings-description"
            className="mb-5 mt-2 text-sm leading-6 text-text-muted"
          >
            将研究、检索和保留策略恢复为默认值。API Key
            和已保存的模型连接不会被删除，重置后仍需点击“保存配置”。
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setShowResetConfirm(false)}
              className="flex-1 rounded-md border border-border px-4 py-2.5 text-sm font-medium hover:bg-surface-alt"
            >
              取消
            </button>
            <button
              type="button"
              onClick={() => {
                resetConfigDefaults();
                setShowResetConfirm(false);
              }}
              className="flex-1 rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-dark"
            >
              确认重置
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
