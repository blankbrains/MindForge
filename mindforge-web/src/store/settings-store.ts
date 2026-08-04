import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

export const LLM_PROVIDERS = [
  "openai",
  "deepseek",
  "kimi",
  "glm",
  "openai_compatible",
  "local",
] as const;

export type LLMProvider = (typeof LLM_PROVIDERS)[number];
export type ResearchMode = "fast" | "balanced" | "deep";
export type SourcePolicy = "auto" | "knowledge_base" | "web";
export type EmbeddingProvider = "openai" | "bge";
export type NativeWebSearchProtocol =
  | "none"
  | "openai_responses"
  | "kimi_builtin"
  | "glm_web_search";

export interface LLMProviderConfig {
  provider: LLMProvider;
  label: string;
  baseUrl: string;
  apiKey: string;
  apiKeyRequired: boolean;
  defaultModel: string;
  plannerModel: string;
  researcherModel: string;
  criticModel: string;
  synthesizerModel: string;
  supportsTools: boolean;
  supportsJsonMode: boolean;
  supportsJsonSchema: boolean;
  nativeWebSearchProtocol: NativeWebSearchProtocol;
  nativeWebSearchEndpoint: string;
  configured: boolean;
}

type ProviderConfigs = Record<LLMProvider, LLMProviderConfig>;
type EditableProviderConfig = Omit<
  LLMProviderConfig,
  "provider" | "label" | "configured"
>;

interface ProviderConfigPayload {
  provider: LLMProvider;
  label: string;
  base_url: string;
  api_key: string;
  api_key_required: boolean;
  default_model: string;
  planner_model: string;
  researcher_model: string;
  critic_model: string;
  synthesizer_model: string;
  supports_tools: boolean;
  supports_json_mode: boolean;
  supports_json_schema: boolean;
  native_web_search_protocol: NativeWebSearchProtocol;
  native_web_search_endpoint: string;
  configured: boolean;
}

interface ProviderUpdatePayload {
  provider: LLMProvider;
  base_url: string;
  api_key: string;
  api_key_required: boolean;
  default_model: string;
  planner_model: string;
  researcher_model: string;
  critic_model: string;
  synthesizer_model: string;
  supports_tools: boolean;
  supports_json_mode: boolean;
  supports_json_schema: boolean;
  native_web_search_protocol: NativeWebSearchProtocol;
  native_web_search_endpoint: string;
}

interface SettingsPayload {
  llm_provider?: LLMProvider;
  llm_configured?: boolean;
  llm_providers?: ProviderConfigPayload[];
  deepseek_api_key?: string;
  openai_api_key?: string;
  compatible_api_key?: string;
  local_api_key?: string;
  llm_provider_configs?: ProviderUpdatePayload[];
  embedding_provider?: EmbeddingProvider;
  research_mode?: ResearchMode;
  source_policy?: SourcePolicy;
  fallback_enabled?: boolean;
  retrieval_top_k?: number;
  rerank_top_k?: number;
  retrieval_min_score?: number;
  keyword_min_coverage?: number;
  max_iterations?: number;
  max_refine_rounds?: number;
  critic_threshold?: number;
  subtask_timeout?: number;
  research_timeout?: number;
  llm_request_timeout?: number;
  max_subtasks?: number;
  max_tool_calls_total?: number;
  max_history_entries?: number;
  langfuse_public_key?: string;
  langfuse_secret_key?: string;
  langfuse_host?: string;
  observability_capture_content?: boolean;
  trace_retention_days?: number;
  tavily_configured?: boolean;
  native_web_search_enabled?: boolean;
  native_web_search_protocol?: NativeWebSearchProtocol;
  native_web_search_supported?: boolean;
  duckduckgo_enabled?: boolean;
  model_only_fallback_enabled?: boolean;
  web_search_available?: boolean;
  reranker_configured?: boolean;
  reranker_available?: boolean;
  reranker_load_failed?: boolean;
}

export interface SettingsState {
  llmProvider: LLMProvider;
  hasLLMKey: boolean;
  providerConfigs: ProviderConfigs;
  savedProviderConfigs: ProviderConfigs;
  dirtyProviders: LLMProvider[];
  embeddingProvider: EmbeddingProvider;
  researchMode: ResearchMode;
  sourcePolicy: SourcePolicy;
  fallbackEnabled: boolean;
  retrievalTopK: number;
  rerankTopK: number;
  retrievalMinScore: number;
  keywordMinCoverage: number;
  maxIterations: number;
  maxRefineRounds: number;
  criticThreshold: number;
  subtaskTimeout: number;
  researchTimeout: number;
  llmRequestTimeout: number;
  maxSubtasks: number;
  maxToolCallsTotal: number;
  maxHistoryEntries: number;
  langfusePublicKey: string;
  langfuseSecretKey: string;
  langfuseHost: string;
  observabilityCaptureContent: boolean;
  traceRetentionDays: number;
  tavilyConfigured: boolean;
  nativeWebSearchEnabled: boolean;
  nativeWebSearchProtocol: NativeWebSearchProtocol;
  nativeWebSearchSupported: boolean;
  duckDuckGoEnabled: boolean;
  modelOnlyFallbackEnabled: boolean;
  webSearchAvailable: boolean;
  rerankerConfigured: boolean;
  rerankerAvailable: boolean;
  rerankerLoadFailed: boolean;
  loaded: boolean;
  loadError: string | null;
  saveError: string | null;

  setLLMProvider: (provider: LLMProvider) => void;
  updateLLMProviderConfig: (
    provider: LLMProvider,
    update: Partial<EditableProviderConfig>,
  ) => void;
  restoreLLMProviderConfig: (provider: LLMProvider) => void;
  setEmbeddingProvider: (value: EmbeddingProvider) => void;
  setResearchMode: (value: ResearchMode) => void;
  setSourcePolicy: (value: SourcePolicy) => void;
  setFallbackEnabled: (value: boolean) => void;
  setRetrievalTopK: (value: number) => void;
  setRerankTopK: (value: number) => void;
  setRetrievalMinScore: (value: number) => void;
  setKeywordMinCoverage: (value: number) => void;
  setMaxIterations: (value: number) => void;
  setMaxRefineRounds: (value: number) => void;
  setCriticThreshold: (value: number) => void;
  setSubtaskTimeout: (value: number) => void;
  setResearchTimeout: (value: number) => void;
  setLLMRequestTimeout: (value: number) => void;
  setMaxSubtasks: (value: number) => void;
  setMaxToolCallsTotal: (value: number) => void;
  setMaxHistoryEntries: (value: number) => void;
  setLangfusePublicKey: (value: string) => void;
  setLangfuseSecretKey: (value: string) => void;
  setLangfuseHost: (value: string) => void;
  setObservabilityCaptureContent: (value: boolean) => void;
  setTraceRetentionDays: (value: number) => void;
  resetConfigDefaults: () => void;
  loadSettings: () => Promise<boolean>;
  saveSettings: () => Promise<boolean>;
  deleteLLMApiKey: (provider?: LLMProvider) => Promise<boolean>;
}

const DEFAULT_PROVIDER_CONFIGS: ProviderConfigs = {
  openai: {
    provider: "openai",
    label: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    apiKey: "",
    apiKeyRequired: true,
    defaultModel: "",
    plannerModel: "gpt-4o",
    researcherModel: "gpt-4o-mini",
    criticModel: "gpt-4o",
    synthesizerModel: "gpt-4o",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: true,
    nativeWebSearchProtocol: "openai_responses",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
  deepseek: {
    provider: "deepseek",
    label: "DeepSeek",
    baseUrl: "https://api.deepseek.com",
    apiKey: "",
    apiKeyRequired: true,
    defaultModel: "",
    plannerModel: "deepseek-v4-flash",
    researcherModel: "deepseek-v4-flash",
    criticModel: "deepseek-v4-flash",
    synthesizerModel: "deepseek-v4-flash",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: false,
    nativeWebSearchProtocol: "openai_responses",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
  kimi: {
    provider: "kimi",
    label: "Kimi",
    baseUrl: "https://api.moonshot.cn/v1",
    apiKey: "",
    apiKeyRequired: true,
    defaultModel: "",
    plannerModel: "",
    researcherModel: "",
    criticModel: "",
    synthesizerModel: "",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: false,
    nativeWebSearchProtocol: "kimi_builtin",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
  glm: {
    provider: "glm",
    label: "GLM",
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    apiKey: "",
    apiKeyRequired: true,
    defaultModel: "",
    plannerModel: "",
    researcherModel: "",
    criticModel: "",
    synthesizerModel: "",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: false,
    nativeWebSearchProtocol: "glm_web_search",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
  openai_compatible: {
    provider: "openai_compatible",
    label: "通用接口",
    baseUrl: "",
    apiKey: "",
    apiKeyRequired: true,
    defaultModel: "",
    plannerModel: "",
    researcherModel: "",
    criticModel: "",
    synthesizerModel: "",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: false,
    nativeWebSearchProtocol: "none",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
  local: {
    provider: "local",
    label: "本地模型服务",
    baseUrl: "http://host.docker.internal:11434/v1",
    apiKey: "",
    apiKeyRequired: false,
    defaultModel: "",
    plannerModel: "",
    researcherModel: "",
    criticModel: "",
    synthesizerModel: "",
    supportsTools: true,
    supportsJsonMode: true,
    supportsJsonSchema: false,
    nativeWebSearchProtocol: "none",
    nativeWebSearchEndpoint: "",
    configured: false,
  },
};

function cloneProviderConfigs(configs: ProviderConfigs): ProviderConfigs {
  return Object.fromEntries(
    LLM_PROVIDERS.map((provider) => [
      provider,
      { ...configs[provider] },
    ]),
  ) as ProviderConfigs;
}

function isLLMProvider(value: unknown): value is LLMProvider {
  return LLM_PROVIDERS.includes(value as LLMProvider);
}

function providerFromPayload(
  payload: ProviderConfigPayload,
): LLMProviderConfig {
  return {
    provider: payload.provider,
    label: payload.label,
    baseUrl: payload.base_url,
    apiKey: payload.api_key,
    apiKeyRequired: payload.api_key_required,
    defaultModel: payload.default_model,
    plannerModel: payload.planner_model,
    researcherModel: payload.researcher_model,
    criticModel: payload.critic_model,
    synthesizerModel: payload.synthesizer_model,
    supportsTools: payload.supports_tools,
    supportsJsonMode: payload.supports_json_mode,
    supportsJsonSchema: payload.supports_json_schema,
    nativeWebSearchProtocol:
      payload.native_web_search_protocol ?? "none",
    nativeWebSearchEndpoint:
      payload.native_web_search_endpoint ?? "",
    configured: payload.configured,
  };
}

function providerToPayload(
  config: LLMProviderConfig,
): ProviderUpdatePayload {
  return {
    provider: config.provider,
    base_url: config.baseUrl,
    api_key: config.apiKey,
    api_key_required: config.apiKeyRequired,
    default_model: config.defaultModel,
    planner_model: config.plannerModel,
    researcher_model: config.researcherModel,
    critic_model: config.criticModel,
    synthesizer_model: config.synthesizerModel,
    supports_tools: config.supportsTools,
    supports_json_mode: config.supportsJsonMode,
    supports_json_schema: config.supportsJsonSchema,
    native_web_search_protocol: config.nativeWebSearchProtocol,
    native_web_search_endpoint: config.nativeWebSearchEndpoint,
  };
}

async function responseError(response: Response, fallback: string) {
  const raw = await response.text().catch(() => "");
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (parsed.detail) return String(parsed.detail);
  } catch {
    // Use the bounded raw response below.
  }
  return raw && raw.length <= 300 ? raw : fallback;
}

function validateSettings(state: SettingsState): string | null {
  const checks: Array<[number, number, number, string]> = [
    [state.retrievalTopK, 1, 100, "向量检索 Top-K"],
    [state.rerankTopK, 1, 50, "重排序 Top-K"],
    [state.retrievalMinScore, 0, 1, "语义相关性阈值"],
    [state.keywordMinCoverage, 0, 1, "关键词覆盖阈值"],
    [state.maxIterations, 1, 20, "最大迭代次数"],
    [state.maxRefineRounds, 0, 5, "最大精炼轮次"],
    [state.criticThreshold, 0, 10, "评判阈值"],
    [state.subtaskTimeout, 10, 600, "子任务超时"],
    [state.researchTimeout, 30, 3600, "研究总超时"],
    [state.llmRequestTimeout, 5, 600, "单次模型调用超时"],
    [state.maxSubtasks, 1, 20, "最大子任务数"],
    [state.maxToolCallsTotal, 1, 100, "工具调用总预算"],
    [state.maxHistoryEntries, 0, 100_000, "历史记录上限"],
    [state.traceRetentionDays, 0, 3650, "Trace 保留天数"],
  ];
  for (const [value, minimum, maximum, label] of checks) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      return `${label}必须在 ${minimum} 到 ${maximum} 之间。`;
    }
  }
  if (state.llmRequestTimeout > state.subtaskTimeout) {
    return "单次模型调用超时不能大于子任务超时。";
  }
  if (state.subtaskTimeout > state.researchTimeout) {
    return "子任务超时不能大于研究总超时。";
  }
  return null;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      llmProvider: "deepseek",
      hasLLMKey: false,
      providerConfigs: cloneProviderConfigs(DEFAULT_PROVIDER_CONFIGS),
      savedProviderConfigs: cloneProviderConfigs(DEFAULT_PROVIDER_CONFIGS),
      dirtyProviders: [],
      embeddingProvider: "bge",
      researchMode: "balanced",
      sourcePolicy: "auto",
      fallbackEnabled: true,
      retrievalTopK: 20,
      rerankTopK: 6,
      retrievalMinScore: 0.6,
      keywordMinCoverage: 0.6,
      maxIterations: 3,
      maxRefineRounds: 1,
      criticThreshold: 7,
      subtaskTimeout: 60,
      researchTimeout: 180,
      llmRequestTimeout: 45,
      maxSubtasks: 5,
      maxToolCallsTotal: 12,
      maxHistoryEntries: 0,
      langfusePublicKey: "",
      langfuseSecretKey: "",
      langfuseHost: "https://cloud.langfuse.com",
      observabilityCaptureContent: false,
      traceRetentionDays: 0,
      tavilyConfigured: false,
      nativeWebSearchEnabled: true,
      nativeWebSearchProtocol: "none",
      nativeWebSearchSupported: false,
      duckDuckGoEnabled: false,
      modelOnlyFallbackEnabled: true,
      webSearchAvailable: false,
      rerankerConfigured: false,
      rerankerAvailable: false,
      rerankerLoadFailed: false,
      loaded: false,
      loadError: null,
      saveError: null,

      setLLMProvider: (provider) => {
        const configured = get().providerConfigs[provider].configured;
        set({
          llmProvider: provider,
          hasLLMKey: configured,
        });
      },

      updateLLMProviderConfig: (provider, update) =>
        set((state) => {
          const nextConfig = {
            ...state.providerConfigs[provider],
            ...update,
          };
          const providerConfigs = {
            ...state.providerConfigs,
            [provider]: nextConfig,
          };
          const isDirty =
            JSON.stringify(providerToPayload(nextConfig)) !==
            JSON.stringify(
              providerToPayload(state.savedProviderConfigs[provider]),
            );
          const dirtyProviders = isDirty
            ? state.dirtyProviders.includes(provider)
              ? state.dirtyProviders
              : [...state.dirtyProviders, provider]
            : state.dirtyProviders.filter((item) => item !== provider);
          return {
            providerConfigs,
            dirtyProviders,
          };
        }),

      restoreLLMProviderConfig: (provider) =>
        set((state) => {
          const restored = { ...state.savedProviderConfigs[provider] };
          const selectedChanged = provider === state.llmProvider;
          return {
            providerConfigs: {
              ...state.providerConfigs,
              [provider]: restored,
            },
            dirtyProviders: state.dirtyProviders.filter(
              (item) => item !== provider,
            ),
            hasLLMKey: selectedChanged
              ? restored.configured
              : state.hasLLMKey,
          };
        }),

      setEmbeddingProvider: (value) => set({ embeddingProvider: value }),
      setResearchMode: (value) => set({ researchMode: value }),
      setSourcePolicy: (value) => set({ sourcePolicy: value }),
      setFallbackEnabled: (value) => set({ fallbackEnabled: value }),
      setRetrievalTopK: (value) => set({ retrievalTopK: value }),
      setRerankTopK: (value) => set({ rerankTopK: value }),
      setRetrievalMinScore: (value) => set({ retrievalMinScore: value }),
      setKeywordMinCoverage: (value) => set({ keywordMinCoverage: value }),
      setMaxIterations: (value) => set({ maxIterations: value }),
      setMaxRefineRounds: (value) => set({ maxRefineRounds: value }),
      setCriticThreshold: (value) => set({ criticThreshold: value }),
      setSubtaskTimeout: (value) => set({ subtaskTimeout: value }),
      setResearchTimeout: (value) => set({ researchTimeout: value }),
      setLLMRequestTimeout: (value) => set({ llmRequestTimeout: value }),
      setMaxSubtasks: (value) => set({ maxSubtasks: value }),
      setMaxToolCallsTotal: (value) => set({ maxToolCallsTotal: value }),
      setMaxHistoryEntries: (value) => set({ maxHistoryEntries: value }),
      setLangfusePublicKey: (value) => set({ langfusePublicKey: value }),
      setLangfuseSecretKey: (value) => set({ langfuseSecretKey: value }),
      setLangfuseHost: (value) => set({ langfuseHost: value }),
      setObservabilityCaptureContent: (value) =>
        set({ observabilityCaptureContent: value }),
      setTraceRetentionDays: (value) => set({ traceRetentionDays: value }),
      resetConfigDefaults: () =>
        set((state) => {
          const configured = state.providerConfigs.deepseek.configured;
          return {
            llmProvider: "deepseek",
            hasLLMKey: configured,
            embeddingProvider: "bge",
            researchMode: "balanced",
            sourcePolicy: "auto",
            fallbackEnabled: true,
            retrievalTopK: 20,
            rerankTopK: 6,
            retrievalMinScore: 0.6,
            keywordMinCoverage: 0.6,
            maxIterations: 3,
            maxRefineRounds: 1,
            criticThreshold: 7,
            subtaskTimeout: 60,
            researchTimeout: 180,
            llmRequestTimeout: 45,
            maxSubtasks: 5,
            maxToolCallsTotal: 12,
            maxHistoryEntries: 0,
            observabilityCaptureContent: false,
            traceRetentionDays: 0,
            saveError: null,
          };
        }),

      loadSettings: async () => {
        set({ loaded: false, loadError: null });
        try {
          const response = await fetch(`${API_BASE}/settings`);
          if (!response.ok) {
            throw new Error(`Settings request failed with ${response.status}`);
          }

          const data = (await response.json()) as SettingsPayload;
          const provider = isLLMProvider(data.llm_provider)
            ? data.llm_provider
            : "deepseek";
          const providerConfigs = cloneProviderConfigs(
            DEFAULT_PROVIDER_CONFIGS,
          );

          for (const item of data.llm_providers ?? []) {
            if (isLLMProvider(item.provider)) {
              providerConfigs[item.provider] = providerFromPayload(item);
            }
          }

          if (!data.llm_providers?.length) {
            providerConfigs.openai.apiKey = data.openai_api_key ?? "";
            providerConfigs.deepseek.apiKey = data.deepseek_api_key ?? "";
            providerConfigs.openai_compatible.apiKey =
              data.compatible_api_key ?? "";
            providerConfigs.local.apiKey = data.local_api_key ?? "";
            for (const item of LLM_PROVIDERS) {
              providerConfigs[item].configured =
                providerConfigs[item].apiKey.length > 0;
            }
          }

          const configured =
            data.llm_configured ?? providerConfigs[provider].configured;
          const savedProviderConfigs = cloneProviderConfigs(providerConfigs);
          set({
            llmProvider: provider,
            hasLLMKey: configured,
            providerConfigs,
            savedProviderConfigs,
            dirtyProviders: [],
            embeddingProvider: data.embedding_provider ?? "bge",
            researchMode: data.research_mode ?? "balanced",
            sourcePolicy: data.source_policy ?? "auto",
            fallbackEnabled: data.fallback_enabled ?? true,
            retrievalTopK: data.retrieval_top_k ?? 20,
            rerankTopK: data.rerank_top_k ?? 6,
            retrievalMinScore: data.retrieval_min_score ?? 0.6,
            keywordMinCoverage: data.keyword_min_coverage ?? 0.6,
            maxIterations: data.max_iterations ?? 3,
            maxRefineRounds: data.max_refine_rounds ?? 1,
            criticThreshold: data.critic_threshold ?? 7,
            subtaskTimeout: data.subtask_timeout ?? 60,
            researchTimeout: data.research_timeout ?? 180,
            llmRequestTimeout: data.llm_request_timeout ?? 45,
            maxSubtasks: data.max_subtasks ?? 5,
            maxToolCallsTotal: data.max_tool_calls_total ?? 12,
            maxHistoryEntries: data.max_history_entries ?? 0,
            langfusePublicKey: data.langfuse_public_key ?? "",
            langfuseSecretKey: data.langfuse_secret_key ?? "",
            langfuseHost:
              data.langfuse_host ?? "https://cloud.langfuse.com",
            observabilityCaptureContent:
              data.observability_capture_content ?? false,
            traceRetentionDays: data.trace_retention_days ?? 0,
            tavilyConfigured: data.tavily_configured ?? false,
            nativeWebSearchEnabled:
              data.native_web_search_enabled ?? true,
            nativeWebSearchProtocol:
              data.native_web_search_protocol ?? "none",
            nativeWebSearchSupported:
              data.native_web_search_supported ?? false,
            duckDuckGoEnabled: data.duckduckgo_enabled ?? false,
            modelOnlyFallbackEnabled:
              data.model_only_fallback_enabled ?? true,
            webSearchAvailable: data.web_search_available ?? false,
            rerankerConfigured: data.reranker_configured ?? false,
            rerankerAvailable: data.reranker_available ?? false,
            rerankerLoadFailed: data.reranker_load_failed ?? false,
            loaded: true,
            loadError: null,
          });
          return true;
        } catch (error) {
          set({
            loaded: true,
            loadError:
              error instanceof Error
                ? error.message
                : "设置加载失败。",
          });
          return false;
        }
      },

      saveSettings: async () => {
        const state = get();
        const validationError = validateSettings(state);
        if (validationError) {
          set({ saveError: validationError });
          return false;
        }
        set({ saveError: null });
        const payload: SettingsPayload = {
          llm_provider: state.llmProvider,
          llm_provider_configs: state.dirtyProviders.map((provider) =>
            providerToPayload(state.providerConfigs[provider]),
          ),
          embedding_provider: state.embeddingProvider,
          research_mode: state.researchMode,
          source_policy: state.sourcePolicy,
          fallback_enabled: state.fallbackEnabled,
          retrieval_top_k: state.retrievalTopK,
          rerank_top_k: state.rerankTopK,
          retrieval_min_score: state.retrievalMinScore,
          keyword_min_coverage: state.keywordMinCoverage,
          max_iterations: state.maxIterations,
          max_refine_rounds: state.maxRefineRounds,
          critic_threshold: state.criticThreshold,
          subtask_timeout: state.subtaskTimeout,
          research_timeout: state.researchTimeout,
          llm_request_timeout: state.llmRequestTimeout,
          max_subtasks: state.maxSubtasks,
          max_tool_calls_total: state.maxToolCallsTotal,
          max_history_entries: state.maxHistoryEntries,
          langfuse_public_key: state.langfusePublicKey,
          langfuse_secret_key: state.langfuseSecretKey,
          langfuse_host: state.langfuseHost,
          observability_capture_content:
            state.observabilityCaptureContent,
          trace_retention_days: state.traceRetentionDays,
        };

        try {
          const response = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            set({
              saveError: await responseError(
                response,
                "设置保存失败，请稍后重试。",
              ),
            });
            return false;
          }
          const reloaded = await get().loadSettings();
          if (!reloaded) {
            set({
              saveError: "设置已保存，但重新加载失败，请刷新页面确认。",
            });
          }
          return reloaded;
        } catch {
          set({ saveError: "设置保存失败，请检查网络连接。" });
          return false;
        }
      },

      deleteLLMApiKey: async (providerOverride) => {
        const provider = providerOverride ?? get().llmProvider;
        const payload: SettingsPayload = {
          llm_provider_configs: [
            {
              ...providerToPayload(get().providerConfigs[provider]),
              api_key: "",
            },
          ],
        };
        try {
          const response = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) return false;
          return await get().loadSettings();
        } catch {
          return false;
        }
      },
    }),
    {
      name: "mindforge-settings-v3",
      partialize: (state) => ({
        llmProvider: state.llmProvider,
        embeddingProvider: state.embeddingProvider,
        researchMode: state.researchMode,
        sourcePolicy: state.sourcePolicy,
        fallbackEnabled: state.fallbackEnabled,
        retrievalTopK: state.retrievalTopK,
        rerankTopK: state.rerankTopK,
        retrievalMinScore: state.retrievalMinScore,
        keywordMinCoverage: state.keywordMinCoverage,
        maxIterations: state.maxIterations,
        maxRefineRounds: state.maxRefineRounds,
        criticThreshold: state.criticThreshold,
        subtaskTimeout: state.subtaskTimeout,
        researchTimeout: state.researchTimeout,
        llmRequestTimeout: state.llmRequestTimeout,
        maxSubtasks: state.maxSubtasks,
        maxToolCallsTotal: state.maxToolCallsTotal,
        maxHistoryEntries: state.maxHistoryEntries,
        langfuseHost: state.langfuseHost,
        observabilityCaptureContent: state.observabilityCaptureContent,
        traceRetentionDays: state.traceRetentionDays,
      }),
    },
  ),
);
