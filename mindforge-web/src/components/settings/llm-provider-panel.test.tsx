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
  configured: true,
};

describe("LLMProviderPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    const current = useSettingsStore.getState().providerConfigs;
    const providerConfigs = {
      ...current,
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

  it("shows the explicit OpenAI Base URL", () => {
    render(<LLMProviderPanel />);

    expect(
      (screen.getByLabelText("Base URL") as HTMLInputElement).value,
    ).toBe("https://api.openai.com/v1");
  });

  it("loads provider models and routes agents from the returned list", async () => {
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

    fireEvent.click(screen.getByRole("button", { name: "拉取模型" }));

    expect(
      await screen.findByText("已从接口加载 2 个模型"),
    ).not.toBeNull();
    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      api_key: "",
      api_key_required: true,
      use_stored_api_key: true,
    });

    fireEvent.change(screen.getByLabelText("Planner"), {
      target: { value: "gpt-4.1" },
    });
    expect(
      useSettingsStore.getState().providerConfigs.openai.plannerModel,
    ).toBe("gpt-4.1");
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

    fireEvent.click(screen.getByRole("button", { name: "拉取模型" }));
    await screen.findByText("已从接口加载 1 个模型");

    fireEvent.change(screen.getByLabelText("Researcher"), {
      target: { value: "__mindforge_custom_model__" },
    });
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

    fireEvent.click(screen.getByRole("button", { name: "拉取模型" }));
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
        screen.queryByText("已从接口加载 1 个模型"),
      ).toBeNull();
    });
  });
});
