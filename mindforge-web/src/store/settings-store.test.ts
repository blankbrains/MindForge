import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  useSettingsStore,
  type LLMProvider,
  type LLMProviderConfig,
} from "@/store/settings-store";

function providerConfig(
  provider: LLMProvider,
  update: Partial<LLMProviderConfig> = {},
): LLMProviderConfig {
  const defaults: Record<LLMProvider, LLMProviderConfig> = {
    openai: {
      provider: "openai",
      label: "OpenAI",
      baseUrl: "",
      apiKey: "***open",
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
      configured: true,
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
      label: "OpenAI 兼容云 API",
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
  return { ...defaults[provider], ...update };
}

function apiProvider(config: LLMProviderConfig) {
  return {
    provider: config.provider,
    label: config.label,
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
    configured: config.configured,
  };
}

function settingsResponse(
  provider: LLMProvider = "openai",
  configs = {
    openai: providerConfig("openai"),
    deepseek: providerConfig("deepseek"),
    kimi: providerConfig("kimi"),
    glm: providerConfig("glm"),
    openai_compatible: providerConfig("openai_compatible"),
    local: providerConfig("local"),
  },
) {
  return new Response(
    JSON.stringify({
      llm_provider: provider,
      llm_configured: configs[provider].configured,
      llm_providers: Object.values(configs).map(apiProvider),
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}

describe("settings store", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    const providerConfigs = {
      openai: providerConfig("openai"),
      deepseek: providerConfig("deepseek"),
      kimi: providerConfig("kimi"),
      glm: providerConfig("glm"),
      openai_compatible: providerConfig("openai_compatible"),
      local: providerConfig("local"),
    };
    useSettingsStore.setState({
      llmProvider: "openai",
      hasLLMKey: true,
      providerConfigs,
      savedProviderConfigs: structuredClone(providerConfigs),
      dirtyProviders: [],
      embeddingProvider: "bge",
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
      loaded: true,
      loadError: null,
      saveError: null,
    });
  });

  it("does not couple the LLM provider to the embedding provider", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(settingsResponse());

    expect(await useSettingsStore.getState().saveSettings()).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body)) as Record<
      string,
      unknown
    >;
    expect(payload.llm_provider).toBe("openai");
    expect(payload.embedding_provider).toBe("bge");
    expect(useSettingsStore.getState().embeddingProvider).toBe("bge");
  });

  it("does not report success when the saved settings cannot be reloaded", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }));

    expect(await useSettingsStore.getState().saveSettings()).toBe(false);
    expect(useSettingsStore.getState().loadError).toContain("503");
    expect(useSettingsStore.getState().saveError).toContain("重新加载失败");
  });

  it("rejects invalid numeric settings before sending a request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    useSettingsStore.setState({ retrievalTopK: 0 });

    expect(await useSettingsStore.getState().saveSettings()).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useSettingsStore.getState().saveError).toContain(
      "向量检索 Top-K",
    );
  });

  it("preserves and saves drafts for multiple providers", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(settingsResponse("local"));

    useSettingsStore.getState().setLLMProvider("openai_compatible");
    useSettingsStore.getState().updateLLMProviderConfig(
      "openai_compatible",
      {
        baseUrl: "https://cloud.example/v1",
        apiKey: "cloud-key",
        defaultModel: "cloud-model",
        nativeWebSearchProtocol: "kimi_builtin",
      },
    );
    useSettingsStore.getState().setLLMProvider("local");
    useSettingsStore.getState().updateLLMProviderConfig("local", {
      defaultModel: "qwen3",
      supportsJsonSchema: true,
    });

    expect(await useSettingsStore.getState().saveSettings()).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body)) as {
      llm_provider: string;
      llm_provider_configs: Array<Record<string, unknown>>;
    };
    expect(payload.llm_provider).toBe("local");
    expect(payload.llm_provider_configs).toHaveLength(2);
    expect(payload.llm_provider_configs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          provider: "openai_compatible",
          api_key: "cloud-key",
          default_model: "cloud-model",
          native_web_search_protocol: "kimi_builtin",
        }),
        expect.objectContaining({
          provider: "local",
          default_model: "qwen3",
          supports_json_schema: true,
        }),
      ]),
    );
  });

  it("keeps Kimi, GLM and generic compatible settings isolated", () => {
    useSettingsStore.getState().updateLLMProviderConfig("kimi", {
      apiKey: "kimi-key",
      defaultModel: "kimi-k2",
    });
    useSettingsStore.getState().updateLLMProviderConfig("glm", {
      apiKey: "glm-key",
      defaultModel: "glm-4.5",
    });
    useSettingsStore.getState().updateLLMProviderConfig(
      "openai_compatible",
      {
        baseUrl: "https://gateway.example/v1",
        apiKey: "generic-key",
        defaultModel: "generic-model",
      },
    );

    const configs = useSettingsStore.getState().providerConfigs;
    expect(configs.kimi.apiKey).toBe("kimi-key");
    expect(configs.kimi.defaultModel).toBe("kimi-k2");
    expect(configs.glm.apiKey).toBe("glm-key");
    expect(configs.glm.defaultModel).toBe("glm-4.5");
    expect(configs.openai_compatible.apiKey).toBe("generic-key");
    expect(configs.openai_compatible.defaultModel).toBe(
      "generic-model",
    );
    expect(useSettingsStore.getState().dirtyProviders).toEqual(
      expect.arrayContaining(["kimi", "glm", "openai_compatible"]),
    );
  });

  it("loads a configured local provider without an API key", async () => {
    const local = providerConfig("local", {
      configured: true,
      defaultModel: "qwen3",
    });
    const configs = {
      openai: providerConfig("openai"),
      deepseek: providerConfig("deepseek"),
      kimi: providerConfig("kimi"),
      glm: providerConfig("glm"),
      openai_compatible: providerConfig("openai_compatible"),
      local,
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      settingsResponse("local", configs),
    );

    await useSettingsStore.getState().loadSettings();

    const state = useSettingsStore.getState();
    expect(state.llmProvider).toBe("local");
    expect(state.hasLLMKey).toBe(true);
    expect(state.providerConfigs.local.apiKey).toBe("");
  });

  it("deletes only the selected provider API key", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(settingsResponse());

    expect(
      await useSettingsStore.getState().deleteLLMApiKey("openai"),
    ).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({
      llm_provider_configs: [
        expect.objectContaining({
          provider: "openai",
          api_key: "",
        }),
      ],
    });
  });

  it("does not report API-key deletion success when reload fails", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }));

    expect(
      await useSettingsStore.getState().deleteLLMApiKey("openai"),
    ).toBe(false);
  });

  it("resets numeric defaults without discarding provider drafts", () => {
    useSettingsStore.getState().updateLLMProviderConfig("local", {
      defaultModel: "qwen3",
    });
    useSettingsStore.setState({
      retrievalTopK: 99,
      maxIterations: 18,
    });

    useSettingsStore.getState().resetConfigDefaults();

    const state = useSettingsStore.getState();
    expect(state.llmProvider).toBe("deepseek");
    expect(state.providerConfigs.local.defaultModel).toBe("qwen3");
    expect(state.dirtyProviders).toContain("local");
    expect(state.retrievalTopK).toBe(20);
    expect(state.maxIterations).toBe(3);
    expect(state.subtaskTimeout).toBe(60);
  });

  it("rejects contradictory timeout budgets", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    useSettingsStore.setState({
      llmRequestTimeout: 90,
      subtaskTimeout: 60,
      researchTimeout: 180,
    });

    expect(await useSettingsStore.getState().saveSettings()).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useSettingsStore.getState().saveError).toContain(
      "单次模型调用超时不能大于子任务超时",
    );
  });
});
