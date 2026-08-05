import { useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import {
  Bot,
  Boxes,
  Cable,
  Check,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Eye,
  EyeOff,
  KeyRound,
  MoonStar,
  RefreshCw,
  RotateCcw,
  Server,
  Sparkles,
  Trash2,
  type LucideIcon,
} from "lucide-react";
import { Tooltip } from "@/components/shared/tooltip";
import { API_BASE } from "@/lib/constants";
import {
  useSettingsStore,
  type LLMProvider,
  type LLMProviderConfig,
  type NativeWebSearchProtocol,
} from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20";

interface ProviderPresetDefinition {
  id: LLMProvider;
  label: string;
  description: string;
  icon: LucideIcon;
}

const PROVIDER_PRESETS: ProviderPresetDefinition[] = [
  {
    id: "openai",
    label: "OpenAI",
    description: "原生 API",
    icon: Sparkles,
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    description: "原生 API",
    icon: Bot,
  },
  {
    id: "kimi",
    label: "Kimi",
    description: "Moonshot API",
    icon: MoonStar,
  },
  {
    id: "glm",
    label: "GLM",
    description: "智谱开放平台",
    icon: Boxes,
  },
  {
    id: "openai_compatible",
    label: "通用接口",
    description: "OpenAI-compatible",
    icon: Cable,
  },
  {
    id: "local",
    label: "本地模型",
    description: "Ollama、vLLM 等",
    icon: Cpu,
  },
];

interface DiscoveredModel {
  id: string;
  ownedBy: string;
}

interface ModelCatalogState {
  status: "idle" | "loading" | "success" | "error";
  models: DiscoveredModel[];
  error: string;
  truncated: boolean;
}

function emptyModelCatalog(): ModelCatalogState {
  return {
    status: "idle",
    models: [],
    error: "",
    truncated: false,
  };
}

function createModelCatalogs(): Record<LLMProvider, ModelCatalogState> {
  return {
    openai: emptyModelCatalog(),
    deepseek: emptyModelCatalog(),
    kimi: emptyModelCatalog(),
    glm: emptyModelCatalog(),
    openai_compatible: emptyModelCatalog(),
    local: emptyModelCatalog(),
  };
}

function createModelRequestIds(): Record<LLMProvider, number> {
  return {
    openai: 0,
    deepseek: 0,
    kimi: 0,
    glm: 0,
    openai_compatible: 0,
    local: 0,
  };
}

function presetIsConfigured(
  preset: ProviderPresetDefinition,
  configs: Record<LLMProvider, LLMProviderConfig>,
): boolean {
  return configs[preset.id].configured;
}

async function modelDiscoveryError(response: Response): Promise<string> {
  const raw = await response.text().catch(() => "");
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
  } catch {
    // Fall through to the bounded response below.
  }
  return raw && raw.length <= 300
    ? raw
    : "连接检测失败，请检查接口地址和密钥。";
}

export function LLMProviderPanel() {
  const provider = useSettingsStore((state) => state.llmProvider);
  const configs = useSettingsStore((state) => state.providerConfigs);
  const dirtyProviders = useSettingsStore((state) => state.dirtyProviders);
  const setProvider = useSettingsStore((state) => state.setLLMProvider);
  const updateConfig = useSettingsStore(
    (state) => state.updateLLMProviderConfig,
  );
  const restoreConfig = useSettingsStore(
    (state) => state.restoreLLMProviderConfig,
  );
  const deleteApiKey = useSettingsStore((state) => state.deleteLLMApiKey);
  const [editingKey, setEditingKey] = useState(false);
  const [keyBeforeEdit, setKeyBeforeEdit] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [showAdvancedRouting, setShowAdvancedRouting] = useState(false);
  const [showAdvancedInterface, setShowAdvancedInterface] = useState(false);
  const [modelCatalogs, setModelCatalogs] = useState(createModelCatalogs);
  const modelRequestIds = useRef(createModelRequestIds());

  const config = configs[provider];
  const catalog = modelCatalogs[provider];
  const isCustomProvider =
    provider === "kimi" ||
    provider === "glm" ||
    provider === "openai_compatible" ||
    provider === "local";
  const isDirty = dirtyProviders.includes(provider);
  const primaryModel = isCustomProvider
    ? config.defaultModel
    : config.researcherModel;

  const invalidateModelCatalog = (target: LLMProvider) => {
    modelRequestIds.current[target] += 1;
    setModelCatalogs((current) => ({
      ...current,
      [target]: emptyModelCatalog(),
    }));
  };

  const updateConnectionConfig = (
    target: LLMProvider,
    update: Partial<LLMProviderConfig>,
  ) => {
    invalidateModelCatalog(target);
    updateConfig(target, update);
  };

  const resetTransientState = () => {
    setEditingKey(false);
    setShowKey(false);
    setDeleteError("");
    setShowAdvancedRouting(false);
    setShowAdvancedInterface(false);
  };

  const selectProvider = (nextProvider: LLMProvider) => {
    if (nextProvider === provider) return;
    if (editingKey) {
      updateConnectionConfig(provider, { apiKey: keyBeforeEdit });
    }
    resetTransientState();
    setProvider(nextProvider);
  };

  const startKeyEdit = () => {
    setKeyBeforeEdit(config.apiKey);
    updateConnectionConfig(provider, { apiKey: "" });
    setEditingKey(true);
    setShowKey(false);
  };

  const cancelKeyEdit = () => {
    updateConnectionConfig(provider, { apiKey: keyBeforeEdit });
    setEditingKey(false);
    setShowKey(false);
  };

  const handleDelete = async () => {
    if (dirtyProviders.length > 0) {
      updateConnectionConfig(provider, { apiKey: "" });
      setEditingKey(false);
      setShowKey(false);
      return;
    }
    setDeleting(true);
    setDeleteError("");
    const deleted = await deleteApiKey(provider);
    setDeleting(false);
    if (deleted) {
      invalidateModelCatalog(provider);
      setEditingKey(false);
      setShowKey(false);
    } else {
      setDeleteError("API Key 删除失败，请重试。");
    }
  };

  const fetchModels = async () => {
    const baseUrl = config.baseUrl.trim();
    const useStoredApiKey = config.apiKey.startsWith("***") && !editingKey;
    if (!baseUrl) {
      setModelCatalogs((current) => ({
        ...current,
        [provider]: {
          ...current[provider],
          status: "error",
          error: "请先填写 Base URL。",
        },
      }));
      return;
    }
    if (config.apiKeyRequired && !useStoredApiKey && !config.apiKey.trim()) {
      setModelCatalogs((current) => ({
        ...current,
        [provider]: {
          ...current[provider],
          status: "error",
          error: "请先填写 API Key。",
        },
      }));
      return;
    }

    const requestId = modelRequestIds.current[provider] + 1;
    modelRequestIds.current[provider] = requestId;
    setModelCatalogs((current) => ({
      ...current,
      [provider]: {
        ...current[provider],
        status: "loading",
        error: "",
      },
    }));
    try {
      const response = await fetch(`${API_BASE}/settings/models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          base_url: baseUrl,
          api_key: useStoredApiKey ? "" : config.apiKey,
          api_key_required: config.apiKeyRequired,
          use_stored_api_key: useStoredApiKey,
        }),
      });
      if (!response.ok) {
        throw new Error(await modelDiscoveryError(response));
      }
      const payload = (await response.json()) as {
        models?: Array<{ id?: unknown; owned_by?: unknown }>;
        truncated?: boolean;
      };
      const models = (payload.models ?? [])
        .filter(
          (item): item is { id: string; owned_by?: unknown } =>
            typeof item.id === "string" && item.id.trim().length > 0,
        )
        .map((item) => ({
          id: item.id.trim(),
          ownedBy:
            typeof item.owned_by === "string" ? item.owned_by.trim() : "",
        }));
      setModelCatalogs((current) => {
        if (modelRequestIds.current[provider] !== requestId) {
          return current;
        }
        return {
          ...current,
          [provider]: {
            status: "success",
            models,
            error: "",
            truncated: Boolean(payload.truncated),
          },
        };
      });
    } catch (error) {
      setModelCatalogs((current) => {
        if (modelRequestIds.current[provider] !== requestId) {
          return current;
        }
        return {
          ...current,
          [provider]: {
            ...current[provider],
            status: "error",
            error:
              error instanceof Error
                ? error.message
                : "连接检测失败，请检查接口地址和密钥。",
          },
        };
      });
    }
  };

  const updatePrimaryModel = (value: string) => {
    if (isCustomProvider) {
      updateConfig(provider, { defaultModel: value });
      return;
    }
    updateConfig(provider, {
      plannerModel: value,
      researcherModel: value,
      criticModel: value,
      synthesizerModel: value,
    });
  };

  return (
    <section
      className="space-y-6 rounded-lg border border-border bg-surface p-4 sm:p-6"
      role="tabpanel"
      aria-label="LLM 供应商配置"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-semibold">模型供应商</h2>
          <p className="mt-1 text-sm text-text-muted">
            选择接口后完成连接与模型配置
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {isDirty && (
            <button
              type="button"
              onClick={() => {
                restoreConfig(provider);
                invalidateModelCatalog(provider);
                resetTransientState();
              }}
              className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-text-muted transition hover:bg-surface-alt hover:text-text"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              撤销
            </button>
          )}
          <StatusBadge configured={config.configured} dirty={isDirty} />
        </div>
      </div>

      <ProviderPresetPicker
        selectedProvider={provider}
        configs={configs}
        onSelect={selectProvider}
      />

      <div className="border-t border-border pt-5">
        <SectionHeading icon={Server} title="连接配置" />
        <div className="mt-4 space-y-4">
          <Field
            id="llm-base-url"
            label="Base URL"
            value={config.baseUrl}
            placeholder={
              provider === "local"
                ? "http://host.docker.internal:11434/v1"
                : "https://example.com/v1"
            }
            onChange={(value) => {
              updateConnectionConfig(provider, { baseUrl: value });
            }}
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
            onChange={(apiKey) => {
              updateConnectionConfig(provider, { apiKey });
            }}
          />
          <button
            type="button"
            onClick={() => void fetchModels()}
            disabled={catalog.status === "loading"}
            className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-text transition hover:bg-surface-alt disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
          >
            <RefreshCw
              className={`h-4 w-4 ${
                catalog.status === "loading" ? "animate-spin" : ""
              }`}
              aria-hidden="true"
            />
            {catalog.status === "loading"
              ? "正在检测连接"
              : "检测连接并拉取模型"}
          </button>
          <ModelCatalogMessage catalog={catalog} />
        </div>
      </div>

      <div className="border-t border-border pt-5">
        <SectionHeading icon={KeyRound} title="模型选择" />
        <div className="mt-4">
          <ModelRouteField
            key={`${provider}-primary`}
            id="llm-primary-model"
            label={isCustomProvider ? "默认模型" : "主要模型"}
            value={primaryModel}
            placeholder="模型服务返回的模型 ID"
            models={catalog.models}
            allowEmpty={false}
            onChange={updatePrimaryModel}
          />
        </div>
      </div>

      <AdvancedSection
        id="advanced-model-routing"
        title="高级模型路由"
        description="按 Agent 分配不同模型"
        open={showAdvancedRouting}
        onToggle={() => setShowAdvancedRouting((value) => !value)}
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <ModelRouteField
            key={`${provider}-planner`}
            id="llm-planner-model"
            label="Planner"
            value={config.plannerModel}
            placeholder={isCustomProvider ? "留空继承默认模型" : ""}
            models={catalog.models}
            allowEmpty={isCustomProvider}
            onChange={(plannerModel) =>
              updateConfig(provider, { plannerModel })
            }
          />
          <ModelRouteField
            key={`${provider}-researcher`}
            id="llm-researcher-model"
            label="Researcher"
            value={config.researcherModel}
            placeholder={isCustomProvider ? "留空继承默认模型" : ""}
            models={catalog.models}
            allowEmpty={isCustomProvider}
            onChange={(researcherModel) =>
              updateConfig(provider, { researcherModel })
            }
          />
          <ModelRouteField
            key={`${provider}-critic`}
            id="llm-critic-model"
            label="Critic"
            value={config.criticModel}
            placeholder={isCustomProvider ? "留空继承默认模型" : ""}
            models={catalog.models}
            allowEmpty={isCustomProvider}
            onChange={(criticModel) => updateConfig(provider, { criticModel })}
          />
          <ModelRouteField
            key={`${provider}-synthesizer`}
            id="llm-synthesizer-model"
            label="Synthesizer"
            value={config.synthesizerModel}
            placeholder={isCustomProvider ? "留空继承默认模型" : ""}
            models={catalog.models}
            allowEmpty={isCustomProvider}
            onChange={(synthesizerModel) =>
              updateConfig(provider, { synthesizerModel })
            }
          />
        </div>
      </AdvancedSection>

      {isCustomProvider && (
        <AdvancedSection
          id="advanced-interface-settings"
          title="高级接口设置"
          description="协议与能力兼容选项"
          open={showAdvancedInterface}
          onToggle={() => setShowAdvancedInterface((value) => !value)}
        >
          <div className="divide-y divide-border rounded-md border border-border">
            <CapabilityToggle
              label="需要 API Key"
              checked={config.apiKeyRequired}
              onChange={(apiKeyRequired) =>
                updateConnectionConfig(provider, { apiKeyRequired })
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
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium">
                原生联网协议
              </span>
              <select
                value={config.nativeWebSearchProtocol}
                onChange={(event) =>
                  updateConfig(provider, {
                    nativeWebSearchProtocol: event.target
                      .value as NativeWebSearchProtocol,
                  })
                }
                className={inputClassName}
              >
                <option value="none">不启用</option>
                <option value="openai_responses">
                  OpenAI Responses Web Search
                </option>
                <option value="kimi_builtin">Kimi 内置联网</option>
                <option value="glm_web_search">GLM Web Search API</option>
              </select>
            </label>
            {config.nativeWebSearchProtocol === "glm_web_search" && (
              <Field
                id="llm-native-web-search-endpoint"
                label="联网搜索 Endpoint"
                value={config.nativeWebSearchEndpoint}
                placeholder={`${config.baseUrl.replace(/\/$/, "")}/web_search`}
                onChange={(nativeWebSearchEndpoint) =>
                  updateConnectionConfig(provider, {
                    nativeWebSearchEndpoint,
                  })
                }
              />
            )}
          </div>
        </AdvancedSection>
      )}
    </section>
  );
}

function ProviderPresetPicker({
  selectedProvider,
  configs,
  onSelect,
}: {
  selectedProvider: LLMProvider;
  configs: Record<LLMProvider, LLMProviderConfig>;
  onSelect: (provider: LLMProvider) => void;
}) {
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const moveSelection = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentIndex: number,
  ) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % PROVIDER_PRESETS.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex =
        (currentIndex - 1 + PROVIDER_PRESETS.length) % PROVIDER_PRESETS.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = PROVIDER_PRESETS.length - 1;
    }
    if (nextIndex === null) return;
    event.preventDefault();
    const nextPreset = PROVIDER_PRESETS[nextIndex];
    onSelect(nextPreset.id);
    buttonRefs.current[nextIndex]?.focus();
  };

  return (
    <div
      className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3"
      role="radiogroup"
      aria-label="选择模型供应商"
    >
      {PROVIDER_PRESETS.map((preset, index) => {
        const Icon = preset.icon;
        const selected = preset.id === selectedProvider;
        const configured = presetIsConfigured(preset, configs);
        return (
          <button
            key={preset.id}
            ref={(node) => {
              buttonRefs.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-label={preset.label}
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onSelect(preset.id)}
            onKeyDown={(event) => moveSelection(event, index)}
            className={`group flex min-h-16 min-w-0 items-center gap-3 rounded-md border px-3 py-2.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 ${
              selected
                ? "border-primary bg-primary text-white shadow-sm"
                : "border-border bg-surface-alt text-text hover:border-primary/50 hover:bg-surface"
            }`}
          >
            <span
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-md ${
                selected
                  ? "bg-white/15 text-white"
                  : "bg-surface text-text-muted group-hover:text-primary"
              }`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold">
                {preset.label}
              </span>
              <span
                className={`mt-0.5 block truncate text-xs ${
                  selected ? "text-white/80" : "text-text-muted"
                }`}
              >
                {preset.description}
              </span>
            </span>
            {selected ? (
              <Check className="h-4 w-4 shrink-0" aria-hidden="true" />
            ) : (
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${
                  configured ? "bg-emerald-500" : "bg-border"
                }`}
                aria-hidden="true"
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

function ModelCatalogMessage({ catalog }: { catalog: ModelCatalogState }) {
  if (catalog.status === "success") {
    return (
      <p
        className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400"
        role="status"
        aria-live="polite"
      >
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        连接成功，已加载 {catalog.models.length} 个模型
        {catalog.truncated ? "，列表已按上限截断" : ""}
      </p>
    );
  }
  if (catalog.status === "error") {
    return (
      <p className="text-xs text-red-600" role="alert">
        {catalog.error}
      </p>
    );
  }
  return null;
}

function AdvancedSection({
  id,
  title,
  description,
  open,
  onToggle,
  children,
}: {
  id: string;
  title: string;
  description: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className="border-t border-border pt-2">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={onToggle}
        className="flex min-h-12 w-full items-center justify-between gap-4 rounded-md px-2 text-left transition hover:bg-surface-alt focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        <span className="min-w-0">
          <span className="block text-sm font-semibold">{title}</span>
          <span className="mt-0.5 block text-xs text-text-muted">
            {description}
          </span>
        </span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-text-muted transition-transform ${
            open ? "rotate-180" : ""
          }`}
          aria-hidden="true"
        />
      </button>
      {open && (
        <div id={id} className="px-2 pb-2 pt-4">
          {children}
        </div>
      )}
    </div>
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
  icon: LucideIcon;
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

function ModelRouteField({
  id,
  label,
  value,
  placeholder,
  models,
  allowEmpty,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  placeholder: string;
  models: DiscoveredModel[];
  allowEmpty: boolean;
  onChange: (value: string) => void;
}) {
  const customValue = "__mindforge_custom_model__";
  const inheritValue = "__mindforge_inherit_model__";
  const modelIds = new Set(models.map((model) => model.id));
  const [manualRequested, setManualRequested] = useState(false);

  if (models.length === 0) {
    return (
      <Field
        id={id}
        label={label}
        value={value}
        placeholder={placeholder}
        onChange={onChange}
      />
    );
  }

  const manual =
    manualRequested ||
    Boolean(value && !modelIds.has(value)) ||
    (!allowEmpty && !value);
  const selectedValue = manual
    ? customValue
    : value || (allowEmpty ? inheritValue : customValue);

  return (
    <div>
      <label
        htmlFor={`${id}-select`}
        className="mb-1.5 block text-sm font-medium text-text"
      >
        {label}
      </label>
      <select
        id={`${id}-select`}
        value={selectedValue}
        onChange={(event) => {
          const nextValue = event.target.value;
          if (nextValue === customValue) {
            setManualRequested(true);
            if (modelIds.has(value)) onChange("");
            return;
          }
          setManualRequested(false);
          onChange(nextValue === inheritValue ? "" : nextValue);
        }}
        className={inputClassName}
      >
        {allowEmpty && <option value={inheritValue}>继承默认模型</option>}
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.id}
            {model.ownedBy ? ` · ${model.ownedBy}` : ""}
          </option>
        ))}
        <option value={customValue}>自定义模型 ID…</option>
      </select>
      {selectedValue === customValue && (
        <input
          id={id}
          value={value}
          placeholder={placeholder || "输入自定义模型 ID"}
          autoComplete="off"
          spellCheck={false}
          aria-label={`${label} 自定义模型 ID`}
          onChange={(event) => onChange(event.target.value)}
          className={`${inputClassName} mt-2`}
        />
      )}
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
        <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2">
          <div className={`${inputClassName} min-w-0 truncate text-text-muted`}>
            {config.apiKey}
          </div>
          <button
            type="button"
            onClick={onStartEdit}
            className="rounded-md border border-border px-3 py-2 text-sm font-medium transition hover:bg-surface-alt"
          >
            修改
          </button>
          <Tooltip
            content="从服务端配置中永久删除已保存的 API Key。"
            side="top"
          >
            <button
              type="button"
              onClick={onDelete}
              disabled={deleting}
              className="rounded-md border border-red-200 p-2 text-red-600 transition hover:bg-red-50 disabled:opacity-50 dark:border-red-900 dark:hover:bg-red-950"
              aria-label="删除 API Key"
            >
              <Trash2 className="h-4 w-4" aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      ) : (
        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="relative min-w-0 flex-1">
            <input
              id="llm-api-key"
              type={showKey ? "text" : "password"}
              value={config.apiKey}
              autoComplete="new-password"
              spellCheck={false}
              onChange={(event) => onChange(event.target.value)}
              className={`${inputClassName} pr-10`}
            />
            <Tooltip
              content={
                showKey
                  ? "隐藏正在输入的 API Key"
                  : "临时显示正在输入的 API Key"
              }
              side="top"
              className="absolute right-2 top-1/2 -translate-y-1/2"
            >
              <button
                type="button"
                onClick={onToggleVisibility}
                className="rounded p-1 text-text-muted transition hover:bg-surface hover:text-text"
                aria-label={showKey ? "隐藏 API Key" : "显示 API Key"}
              >
                {showKey ? (
                  <EyeOff className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <Eye className="h-4 w-4" aria-hidden="true" />
                )}
              </button>
            </Tooltip>
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
