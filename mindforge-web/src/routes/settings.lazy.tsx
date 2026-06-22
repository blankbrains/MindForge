import { createLazyRoute } from "@tanstack/react-router";
import { useState } from "react";

type TabId = "llm" | "retrieval" | "agent";

export function SettingsPage() {
  const [tab, setTab] = useState<TabId>("llm");

  const tabs: { id: TabId; label: string }[] = [
    { id: "llm", label: "LLM 供应商" },
    { id: "retrieval", label: "检索配置" },
    { id: "agent", label: "Agent 配置" },
  ];

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">系统配置</h1>
        <p className="mt-1 text-text-muted">管理 LLM、检索与 Agent 参数</p>
      </div>

      <div className="flex gap-1 rounded-xl border border-border bg-surface-alt p-1">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              tab === id
                ? "bg-surface text-text shadow-sm"
                : "text-text-muted hover:text-text"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "llm" && (
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
          <h3 className="font-semibold">LLM 供应商配置</h3>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block text-sm font-medium text-text">
                供应商
              </label>
              <select className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none">
                <option value="openai">OpenAI</option>
                <option value="deepseek">DeepSeek</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-text">
                API Key
              </label>
              <input
                type="password"
                placeholder="sk-..."
                className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none"
              />
            </div>
          </div>
        </div>
      )}

      {tab === "retrieval" && (
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
          <h3 className="font-semibold">检索参数</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text">
                向量检索 Top-K
              </label>
              <input
                type="number"
                defaultValue={20}
                className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text">
                重排序 Top-K
              </label>
              <input
                type="number"
                defaultValue={6}
                className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none"
              />
            </div>
          </div>
        </div>
      )}

      {tab === "agent" && (
        <div className="rounded-xl border border-border bg-surface p-6 space-y-4">
          <h3 className="font-semibold">Agent 参数</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text">
                最大迭代次数
              </label>
              <input
                type="number"
                defaultValue={8}
                className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text">
                评判阈值
              </label>
              <input
                type="number"
                defaultValue={7.0}
                step={0.1}
                className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:outline-none"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const Route = createLazyRoute("/settings")({
  component: SettingsPage,
});
