import { beforeEach, describe, expect, it, vi } from "vitest";

import { useHistoryStore } from "@/store/history-store";

describe("history store", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    useHistoryStore.setState({ entries: [], loaded: false });
  });

  it("does not create a local duplicate after a successful malformed response", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const serverEntry = {
      id: 7,
      task: "research",
      report: null,
      quality_score: 8,
      model_used: "test",
      created_at: "2026-07-29T00:00:00Z",
    };
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response("not-json", { status: 201 }),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ entries: [serverEntry], total: 1 }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      );

    await useHistoryStore.getState().addFromResearch(
      "research",
      "complete report",
      8,
      "test",
    );

    expect(useHistoryStore.getState().entries).toEqual([serverEntry]);
  });

  it("persists token usage and billing status with research history", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: 8 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await useHistoryStore.getState().addFromResearch(
      "research",
      "report",
      8,
      "model",
      {
        tokenUsage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
        costUsd: 0.001,
        costStatus: "estimated",
      },
    );

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(String(request.body)) as {
      token_usage: Record<string, unknown>;
    };
    expect(body.token_usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      billing: {
        estimated_cost_usd: 0.001,
        status: "estimated",
      },
    });
  });

  it("persists only compact, safe citation source metadata", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: 9 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await useHistoryStore.getState().addFromResearch(
      "research",
      "report [1]",
      8,
      "model",
      undefined,
      [
        {
          index: 1,
          title: "Source",
          url: "https://example.com/source",
          source: "web",
          chunk_id: "chunk-1",
          doc_id: "doc-1",
        },
      ],
    );

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(String(request.body)) as {
      sources: Array<Record<string, unknown>>;
    };
    expect(body.sources).toEqual([
      {
        index: 1,
        title: "Source",
        url: "https://example.com/source",
        source: "web",
        chunk_id: "chunk-1",
        doc_id: "doc-1",
      },
    ]);
    expect(body.sources[0]).not.toHaveProperty("content");
  });
});
