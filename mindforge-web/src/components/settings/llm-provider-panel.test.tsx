import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LLMProviderPanel } from "./llm-provider-panel";
import {
  useSettingsStore,
  type LLMProviderConfig,
} from "@/store/settings-store";

const initialProviderConfigs = structuredClone(
  useSettingsStore.getState().providerConfigs,
);

const openAIConfig: LLMProviderConfig = {
  provider: "openai",
  label: "OpenAI",
  baseUrl: "https://api.openai.com/v1",
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
};

describe("LLMProviderPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    const providerConfigs = {
      ...structuredClone(initialProviderConfigs),
      openai: { ...openAIConfig },
    };
    useSettingsStore.setState({
      llmProvider: "openai",
      hasLLMKey: true,
      providerConfigs,
      savedProviderConfigs: structuredClone(providerConfigs),
      dirtyProviders: [],
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("shows six user-facing provider presets and hides advanced fields", () => {
    render(<LLMProviderPanel />);

    expect(screen.getAllByRole("radio")).toHaveLength(6);
    expect(
      screen
        .getByRole("radio", { name: "OpenAI" })
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(screen.getByRole("radio", { name: /DeepSeek/ })).not.toBeNull();
    expect(screen.getByRole("radio", { name: /Kimi/ })).not.toBeNull();
    expect(screen.getByRole("radio", { name: /GLM/ })).not.toBeNull();
    expect(screen.getByRole("radio", { name: /通用接口/ })).not.toBeNull();
    expect(screen.getByRole("radio", { name: /本地模型/ })).not.toBeNull();
    expect(screen.queryByLabelText("Planner")).toBeNull();
    expect(screen.queryByLabelText("原生联网协议")).toBeNull();
    expect(
      screen
        .getByRole("button", { name: /高级模型路由/ })
        .getAttribute("aria-expanded"),
    ).toBe("false");
  });

  it("shows the explicit OpenAI Base URL", () => {
    render(<LLMProviderPanel />);

    expect(
      (screen.getByLabelText("Base URL") as HTMLInputElement).value,
    ).toBe("https://api.openai.com/v1");
  });

  it("supports keyboard selection within the provider preset group", () => {
    render(<LLMProviderPanel />);

    const openAI = screen.getByRole("radio", { name: "OpenAI" });
    fireEvent.keyDown(openAI, { key: "ArrowRight" });

    expect(
      screen
        .getByRole("radio", { name: /DeepSeek/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(useSettingsStore.getState().llmProvider).toBe("deepseek");
  });

  it("selects Kimi without overwriting the generic compatible config", () => {
    useSettingsStore.setState((state) => {
      const compatible = {
        ...state.providerConfigs.openai_compatible,
        baseUrl: "https://gateway.example/v1",
        apiKey: "***compatible",
        defaultModel: "old-model",
        configured: true,
      };
      const providerConfigs = {
        ...state.providerConfigs,
        openai_compatible: compatible,
      };
      return {
        providerConfigs,
        savedProviderConfigs: structuredClone(providerConfigs),
      };
    });
    render(<LLMProviderPanel />);

    fireEvent.click(screen.getByRole("radio", { name: /Kimi/ }));

    const state = useSettingsStore.getState();
    expect(state.llmProvider).toBe("kimi");
    expect(state.providerConfigs.kimi.baseUrl).toBe(
      "https://api.moonshot.cn/v1",
    );
    expect(state.providerConfigs.kimi.nativeWebSearchProtocol).toBe(
      "kimi_builtin",
    );
    expect(state.providerConfigs.openai_compatible.apiKey).toBe(
      "***compatible",
    );
    expect(state.providerConfigs.openai_compatible.defaultModel).toBe(
      "old-model",
    );
  });

  it("applies the GLM endpoint and native search preset", () => {
    render(<LLMProviderPanel />);

    fireEvent.click(screen.getByRole("radio", { name: /GLM/ }));

    const glm = useSettingsStore.getState().providerConfigs.glm;
    expect(useSettingsStore.getState().llmProvider).toBe("glm");
    expect(glm.baseUrl).toBe(
      "https://open.bigmodel.cn/api/paas/v4",
    );
    expect(glm.nativeWebSearchProtocol).toBe("glm_web_search");
  });

  it("loads models, updates the primary route and reveals agent routing", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          models: [
            { id: "gpt-4.1", owned_by: "openai" },
            { id: "gpt-4.1-mini", owned_by: "openai" },
          ],
          count: 2,
          truncated: false,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    render(<LLMProviderPanel />);

    fireEvent.click(
      screen.getByRole("button", {
        name: "检测连接并拉取模型",
      }),
    );

    expect(
      await screen.findByText("连接成功，已加载 2 个模型"),
    ).not.toBeNull();
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      api_key: "",
      api_key_required: true,
      use_stored_api_key: true,
    });

    fireEvent.change(screen.getByLabelText("主要模型"), {
      target: { value: "gpt-4.1" },
    });
    const openAI = useSettingsStore.getState().providerConfigs.openai;
    expect(openAI.plannerModel).toBe("gpt-4.1");
    expect(openAI.researcherModel).toBe("gpt-4.1");
    expect(openAI.criticModel).toBe("gpt-4.1");
    expect(openAI.synthesizerModel).toBe("gpt-4.1");

    fireEvent.click(
      screen.getByRole("button", { name: /高级模型路由/ }),
    );
    fireEvent.change(screen.getByLabelText("Planner"), {
      target: { value: "gpt-4.1-mini" },
    });
    expect(
      useSettingsStore.getState().providerConfigs.openai.plannerModel,
    ).toBe("gpt-4.1-mini");
  });

  it("keeps a manual model input for IDs absent from the catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          models: [{ id: "gpt-4.1", owned_by: "openai" }],
          count: 1,
          truncated: false,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    render(<LLMProviderPanel />);

    fireEvent.click(
      screen.getByRole("button", {
        name: "检测连接并拉取模型",
      }),
    );
    await screen.findByText("连接成功，已加载 1 个模型");
    fireEvent.click(
      screen.getByRole("button", { name: /高级模型路由/ }),
    );

    fireEvent.change(
      screen.getByLabelText("Researcher 自定义模型 ID"),
      {
        target: { value: "fine-tuned-researcher" },
      },
    );

    await waitFor(() => {
      expect(
        useSettingsStore.getState().providerConfigs.openai
          .researcherModel,
      ).toBe("fine-tuned-researcher");
    });
  });

  it("ignores a model response after the connection draft changes", async () => {
    let resolveResponse!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveResponse = resolve;
        }),
    );
    render(<LLMProviderPanel />);

    fireEvent.click(
      screen.getByRole("button", {
        name: "检测连接并拉取模型",
      }),
    );
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: "https://proxy.example/v1" },
    });
    resolveResponse(
      new Response(
        JSON.stringify({
          models: [{ id: "stale-model" }],
          count: 1,
          truncated: false,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await waitFor(() => {
      expect(
        screen.queryByText("连接成功，已加载 1 个模型"),
      ).toBeNull();
    });
  });

  it("reveals protocol controls only from advanced interface settings", () => {
    useSettingsStore.setState((state) => ({
      llmProvider: "openai_compatible",
      providerConfigs: {
        ...state.providerConfigs,
        openai_compatible: {
          ...state.providerConfigs.openai_compatible,
          baseUrl: "https://gateway.example/v1",
        },
      },
    }));
    render(<LLMProviderPanel />);

    expect(screen.queryByLabelText("原生联网协议")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: /高级接口设置/ }),
    );
    fireEvent.change(screen.getByLabelText("原生联网协议"), {
      target: { value: "openai_responses" },
    });

    expect(
      useSettingsStore.getState().providerConfigs.openai_compatible
        .nativeWebSearchProtocol,
    ).toBe("openai_responses");
  });
});
