import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSettingsStore } from "@/store/settings-store";

describe("settings store", () => {
  beforeEach(() => {
    localStorage.clear();
    useSettingsStore.setState({
      llmProvider: "openai",
      llmApiKey: "***1234",
      maskedKeys: { openai: "***1234", deepseek: "" },
      apiKeyDrafts: {},
      hasLLMKey: true,
      hasLLMKeys: { openai: true, deepseek: false },
      retrievalTopK: 20,
      rerankTopK: 6,
      maxIterations: 3,
      maxRefineRounds: 1,
      criticThreshold: 7,
      subtaskTimeout: 30,
      researchTimeout: 180,
      saveError: null,
    });
  });

  it("does not couple the LLM provider to the embedding provider", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            llm_provider: "openai",
            openai_api_key: "***1234",
            embedding_provider: "bge",
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      );

    expect(await useSettingsStore.getState().saveSettings()).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body)) as Record<
      string,
      unknown
    >;
    expect(payload.llm_provider).toBe("openai");
    expect(payload).not.toHaveProperty("embedding_provider");
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

  it("saves API key drafts for both providers after switching", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            llm_provider: "deepseek",
            deepseek_api_key: "***seek",
            openai_api_key: "***open",
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      );

    useSettingsStore.getState().setLLMApiKey("openai-new");
    useSettingsStore.getState().setLLMProvider("deepseek");
    useSettingsStore.getState().setLLMApiKey("deepseek-new");

    expect(await useSettingsStore.getState().saveSettings()).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body)) as Record<
      string,
      unknown
    >;
    expect(payload.openai_api_key).toBe("openai-new");
    expect(payload.deepseek_api_key).toBe("deepseek-new");
  });

  it("deletes an API key without resubmitting unrelated drafts", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            llm_provider: "openai",
            openai_api_key: "",
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      );
    useSettingsStore.setState({ retrievalTopK: 0 });

    expect(await useSettingsStore.getState().deleteLLMApiKey()).toBe(true);

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(request.body)) as Record<
      string,
      unknown
    >;
    expect(payload).toEqual({ openai_api_key: "" });
  });

  it("resets configurable defaults without staging API key deletion", () => {
    useSettingsStore.setState({
      maskedKeys: {
        openai: "***open",
        deepseek: "***seek",
      },
      apiKeyDrafts: {
        openai: "openai-new",
        deepseek: "deepseek-new",
      },
      hasLLMKeys: {
        openai: true,
        deepseek: true,
      },
      retrievalTopK: 99,
      maxIterations: 18,
    });

    useSettingsStore.getState().resetConfigDefaults();

    const state = useSettingsStore.getState();
    expect(state.llmProvider).toBe("deepseek");
    expect(state.llmApiKey).toBe("deepseek-new");
    expect(state.apiKeyDrafts).toEqual({
      openai: "openai-new",
      deepseek: "deepseek-new",
    });
    expect(state.hasLLMKeys).toEqual({
      openai: true,
      deepseek: true,
    });
    expect(state.retrievalTopK).toBe(20);
    expect(state.maxIterations).toBe(3);
  });
});
