import { useState } from "react";
import {
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  RotateCcw,
  Server,
  Trash2,
} from "lucide-react";
import {
  LLM_PROVIDERS,
  useSettingsStore,
  type LLMProvider,
  type LLMProviderConfig,
} from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20";

const providerDescriptions: Record<LLMProvider, string> = {
  openai: "OpenAI 原生接口",
  deepseek: "DeepSeek 原生接口",
  openai_compatible: "通义、Kimi、硅基流动、Gemini 等兼容接口",
  local: "vLLM、Ollama、LM Studio 等本地服务",
};

export function LLMProviderPanel() {
  const provider = useSettingsStore((state) => state.llmProvider);
  const configs = useSettingsStore((state) => state.providerConfigs);
  const dirtyProviders = useSettingsStore(
    (state) => state.dirtyProviders,
  );
  const setProvider = useSettingsStore((state) => state.setLLMProvider);
  const updateConfig = useSettingsStore(
    (state) => state.updateLLMProviderConfig,
  );
  const restoreConfig = useSettingsStore(
    (state) => state.restoreLLMProviderConfig,
  );
  const deleteApiKey = useSettingsStore(
    (state) => state.deleteLLMApiKey,
  );
  const [editingKey, setEditingKey] = useState(false);
  const [keyBeforeEdit, setKeyBeforeEdit] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");

  const config = configs[provider];
  const isCustomProvider =
    provider === "openai_compatible" || provider === "local";
  const isDirty = dirtyProviders.includes(provider);

  const selectProvider = (nextProvider: LLMProvider) => {
    if (editingKey) {
      updateConfig(provider, { apiKey: keyBeforeEdit });
    }
    setEditingKey(false);
    setShowKey(false);
    setDeleteError("");
    setProvider(nextProvider);
  };

  const startKeyEdit = () => {
    setKeyBeforeEdit(config.apiKey);
    updateConfig(provider, { apiKey: "" });
    setEditingKey(true);
    setShowKey(false);
  };

  const cancelKeyEdit = () => {
    updateConfig(provider, { apiKey: keyBeforeEdit });
    setEditingKey(false);
    setShowKey(false);
  };

  const handleDelete = async () => {
    if (dirtyProviders.length > 0) {
      updateConfig(provider, { apiKey: "" });
      setEditingKey(false);
      setShowKey(false);
      return;
    }
    setDeleting(true);
    setDeleteError("");
    const deleted = await deleteApiKey(provider);
    setDeleting(false);
    if (deleted) {
      setEditingKey(false);
      setShowKey(false);
    } else {
      setDeleteError("API Key 删除失败，请重试。");
    }
  };

  return (
    <section
      className="space-y-6 rounded-lg border border-border bg-surface p-5 sm:p-6"
      role="tabpanel"
      aria-label="LLM 供应商配置"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-semibold">模型供应商</h2>
          <p className="mt-1 text-sm text-text-muted">
            {providerDescriptions[provider]}
          </p>
        </div>
        <StatusBadge configured={config.configured} dirty={isDirty} />
      </div>

      <div
        className="grid grid-cols-1 gap-2 sm:grid-cols-2"
        role="radiogroup"
        aria-label="选择模型供应商"
      >
        {LLM_PROVIDERS.map((item) => {
          const itemConfig = configs[item];
          const selected = item === provider;
          return (
            <button
              key={item}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => selectProvider(item)}
              className={`flex min-h-14 items-center justify-between rounded-md border px-3 py-2 text-left transition ${
                selected
                  ? "border-primary bg-primary/5 text-text"
                  : "border-border bg-surface-alt text-text-muted hover:border-text-muted/50 hover:text-text"
              }`}
            >
              <span>
                <span className="block text-sm font-medium">
                  {itemConfig.label}
                </span>
                <span className="mt-0.5 block text-xs">
                  {providerDescriptions[item]}
                </span>
              </span>
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${
                  itemConfig.configured
                    ? "bg-emerald-500"
                    : "bg-border"
                }`}
                aria-hidden="true"
              />
            </button>
          );
        })}
      </div>

      <div className="border-t border-border pt-5">
        <SectionHeading icon={Server} title="连接" />
        <div className="mt-4 space-y-4">
          <Field
            id="llm-base-url"
            label="Base URL"
            value={config.baseUrl}
            placeholder={
              provider === "openai"
                ? "使用 OpenAI 默认地址"
                : "https://example.com/v1"
            }
            onChange={(value) =>
              updateConfig(provider, { baseUrl: value })
            }
          />
          <ApiKeyField
            config={config}
            editing={editingKey}
            showKey={showKey}
            deleting={deleting}
            deleteError={deleteError}
            onStartEdit={startKeyEdit}
            onCancel={cancelKeyEdit}
            onDelete={() => void handleDelete()}
            onToggleVisibility={() => setShowKey((value) => !value)}
            onChange={(apiKey) => updateConfig(provider, { apiKey })}
          />
        </div>
      </div>

      <div className="border-t border-border pt-5">
        <SectionHeading icon={KeyRound} title="模型路由" />
        <div className="mt-4 space-y-4">
          {isCustomProvider && (
            <Field
              id="llm-default-model"
              label="默认模型"
              value={config.defaultModel}
              placeholder="模型服务返回的模型 ID"
              onChange={(defaultModel) =>
                updateConfig(provider, { defaultModel })
              }
            />
          )}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              id="llm-planner-model"
              label="Planner"
              value={config.plannerModel}
              placeholder={isCustomProvider ? "留空继承默认模型" : ""}
              onChange={(plannerModel) =>
                updateConfig(provider, { plannerModel })
              }
            />
            <Field
              id="llm-researcher-model"
              label="Researcher"
              value={config.researcherModel}
              placeholder={isCustomProvider ? "留空继承默认模型" : ""}
              onChange={(researcherModel) =>
                updateConfig(provider, { researcherModel })
              }
            />
            <Field
              id="llm-critic-model"
              label="Critic"
              value={config.criticModel}
              placeholder={isCustomProvider ? "留空继承默认模型" : ""}
              onChange={(criticModel) =>
                updateConfig(provider, { criticModel })
              }
            />
            <Field
              id="llm-synthesizer-model"
              label="Synthesizer"
              value={config.synthesizerModel}
              placeholder={isCustomProvider ? "留空继承默认模型" : ""}
              onChange={(synthesizerModel) =>
                updateConfig(provider, { synthesizerModel })
              }
            />
          </div>
        </div>
      </div>

      {isCustomProvider && (
        <div className="border-t border-border pt-5">
          <div className="flex items-center justify-between gap-4">
            <h3 className="text-sm font-semibold">接口能力</h3>
            {isDirty && (
              <button
                type="button"
                onClick={() => {
                  restoreConfig(provider);
                  setEditingKey(false);
                  setShowKey(false);
                }}
                className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-text-muted transition hover:bg-surface-alt hover:text-text"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                撤销此供应商
              </button>
            )}
          </div>
          <div className="mt-3 divide-y divide-border rounded-md border border-border">
            <CapabilityToggle
              label="需要 API Key"
              checked={config.apiKeyRequired}
              onChange={(apiKeyRequired) =>
                updateConfig(provider, { apiKeyRequired })
              }
            />
            <CapabilityToggle
              label="工具调用"
              checked={config.supportsTools}
              onChange={(supportsTools) =>
                updateConfig(provider, { supportsTools })
              }
            />
            <CapabilityToggle
              label="JSON Mode"
              checked={config.supportsJsonMode}
              onChange={(supportsJsonMode) =>
                updateConfig(provider, { supportsJsonMode })
              }
            />
            <CapabilityToggle
              label="JSON Schema"
              checked={config.supportsJsonSchema}
              onChange={(supportsJsonSchema) =>
                updateConfig(provider, { supportsJsonSchema })
              }
            />
          </div>
        </div>
      )}
    </section>
  );
}

function StatusBadge({
  configured,
  dirty,
}: {
  configured: boolean;
  dirty: boolean;
}) {
  if (dirty) {
    return (
      <span className="rounded-full bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-300">
        待保存
      </span>
    );
  }
  return configured ? (
    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
      <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
      可用
    </span>
  ) : (
    <span className="rounded-full bg-surface-alt px-2.5 py-1 text-xs font-medium text-text-muted">
      未就绪
    </span>
  );
}

function SectionHeading({
  icon: Icon,
  title,
}: {
  icon: typeof Server;
  title: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-4 w-4 text-text-muted" aria-hidden="true" />
      <h3 className="text-sm font-semibold">{title}</h3>
    </div>
  );
}

function Field({
  id,
  label,
  value,
  placeholder,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
}) {
  return (
    <div>
      <label
        htmlFor={id}
        className="mb-1.5 block text-sm font-medium text-text"
      >
        {label}
      </label>
      <input
        id={id}
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
        className={inputClassName}
      />
    </div>
  );
}

function ApiKeyField({
  config,
  editing,
  showKey,
  deleting,
  deleteError,
  onStartEdit,
  onCancel,
  onDelete,
  onToggleVisibility,
  onChange,
}: {
  config: LLMProviderConfig;
  editing: boolean;
  showKey: boolean;
  deleting: boolean;
  deleteError: string;
  onStartEdit: () => void;
  onCancel: () => void;
  onDelete: () => void;
  onToggleVisibility: () => void;
  onChange: (value: string) => void;
}) {
  const maskedKey = config.apiKey.startsWith("***");
  const hasStoredKey = maskedKey && !editing;

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <label htmlFor="llm-api-key" className="text-sm font-medium">
          API Key
          {!config.apiKeyRequired && (
            <span className="ml-1 font-normal text-text-muted">（可选）</span>
          )}
        </label>
        {hasStoredKey && (
          <span className="text-xs text-text-muted">已加密保存</span>
        )}
      </div>
      {hasStoredKey ? (
        <div className="flex items-center gap-2">
          <div className={`${inputClassName} flex-1 text-text-muted`}>
            {config.apiKey}
          </div>
          <button
            type="button"
            onClick={onStartEdit}
            className="rounded-md border border-border px-3 py-2 text-sm font-medium transition hover:bg-surface-alt"
          >
            修改
          </button>
          <button
            type="button"
            onClick={onDelete}
            disabled={deleting}
            className="rounded-md border border-red-200 p-2 text-red-600 transition hover:bg-red-50 disabled:opacity-50 dark:border-red-900 dark:hover:bg-red-950"
            aria-label="删除 API Key"
            title="删除 API Key"
          >
            <Trash2 className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <input
              id="llm-api-key"
              type={showKey ? "text" : "password"}
              value={config.apiKey}
              autoComplete="new-password"
              spellCheck={false}
              onChange={(event) => onChange(event.target.value)}
              className={`${inputClassName} pr-10`}
            />
            <button
              type="button"
              onClick={onToggleVisibility}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-text-muted transition hover:bg-surface hover:text-text"
              aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}
              title={showKey ? "隐藏 API Key" : "显示 API Key"}
            >
              {showKey ? (
                <EyeOff className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Eye className="h-4 w-4" aria-hidden="true" />
              )}
            </button>
          </div>
          {editing && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-md border border-border px-3 py-2 text-sm font-medium text-text-muted transition hover:bg-surface-alt hover:text-text"
            >
              取消
            </button>
          )}
        </div>
      )}
      {deleteError && (
        <p className="mt-2 text-xs text-red-600" role="alert">
          {deleteError}
        </p>
      )}
    </div>
  );
}

function CapabilityToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 px-3 py-2.5">
      <span className="text-sm">{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="peer sr-only"
      />
      <span
        className="relative h-5 w-9 shrink-0 rounded-full bg-border transition peer-checked:bg-primary peer-focus-visible:ring-2 peer-focus-visible:ring-primary/30 peer-focus-visible:ring-offset-2 after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-transform peer-checked:after:translate-x-4"
        aria-hidden="true"
      />
    </label>
  );
}
