import { beforeEach, describe, expect, it, vi } from "vitest";

import { useHistoryStore } from "@/store/history-store";

describe("history store", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    useHistoryStore.setState({
      entries: [],
      localEntries: [],
      loaded: false,
      loading: false,
      loadError: null,
      page: 1,
      pageSize: 20,
      total: 0,
      serverTotal: 0,
    });
  });

  it("loads only the requested server page", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          entries: [
            {
              id: 21,
              task: "page two",
              report: null,
              quality_score: null,
              model_used: null,
              created_at: "2026-07-29T00:00:00Z",
            },
          ],
          total: 45,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await useHistoryStore.getState().loadHistory(2);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "page=2&page_size=20",
    );
    expect(useHistoryStore.getState()).toMatchObject({
      page: 2,
      total: 45,
      serverTotal: 45,
      loaded: true,
      loading: false,
      loadError: null,
    });
  });

  it("surfaces history load failures without discarding local entries", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    const localEntry = {
      id: 1_000_000_000_001,
      task: "local",
      report: "local report",
      quality_score: null,
      model_used: null,
      created_at: "2026-07-29T00:00:00Z",
    };
    useHistoryStore.setState({
      entries: [localEntry],
      localEntries: [localEntry],
    });

    await useHistoryStore.getState().loadHistory(1);

    expect(useHistoryStore.getState()).toMatchObject({
      entries: [localEntry],
      loaded: true,
      loading: false,
      loadError: "offline",
    });
  });

  it("ignores an older page response that finishes last", async () => {
    let resolveFirst!: (response: Response) => void;
    let resolveSecond!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch")
      .mockImplementationOnce(
        () => new Promise<Response>((resolve) => {
          resolveFirst = resolve;
        }),
      )
      .mockImplementationOnce(
        () => new Promise<Response>((resolve) => {
          resolveSecond = resolve;
        }),
      );

    const firstLoad = useHistoryStore.getState().loadHistory(1);
    const secondLoad = useHistoryStore.getState().loadHistory(2);
    resolveSecond(
      new Response(
        JSON.stringify({
          entries: [{
            id: 21,
            task: "new page",
            report: null,
            quality_score: null,
            model_used: null,
            created_at: "2026-07-29T00:00:00Z",
          }],
          total: 40,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    await secondLoad;
    resolveFirst(
      new Response(
        JSON.stringify({
          entries: [{
            id: 1,
            task: "stale page",
            report: null,
            quality_score: null,
            model_used: null,
            created_at: "2026-07-28T00:00:00Z",
          }],
          total: 40,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    await firstLoad;

    expect(useHistoryStore.getState()).toMatchObject({
      page: 2,
      entries: [expect.objectContaining({ task: "new page" })],
      loading: false,
    });
  });

  it("keeps local fallback records visible when a server record is added", () => {
    const localEntry = {
      id: 1_000_000_000_001,
      task: "local fallback",
      report: "local report",
      quality_score: null,
      model_used: null,
      created_at: "2026-07-01T00:00:00Z",
    };
    const serverEntries = Array.from({ length: 20 }, (_, index) => ({
      id: index + 1,
      task: `server ${index + 1}`,
      report: null,
      quality_score: null,
      model_used: null,
      created_at: `2026-07-${String(index + 2).padStart(2, "0")}T00:00:00Z`,
    }));
    useHistoryStore.setState({
      entries: [...serverEntries, localEntry],
      localEntries: [localEntry],
      loaded: true,
      page: 1,
      pageSize: 20,
      total: 21,
      serverTotal: 20,
    });

    useHistoryStore.getState().addEntry({
      id: 100,
      task: "new server record",
      report: null,
      quality_score: 8,
      model_used: "test",
      created_at: "2026-07-31T00:00:00Z",
    });

    const state = useHistoryStore.getState();
    expect(state.entries).toContainEqual(localEntry);
    expect(
      state.entries.filter((entry) => entry.id < 1_000_000_000_000),
    ).toHaveLength(20);
    expect(state.total).toBe(22);
    expect(state.serverTotal).toBe(21);
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

  it("persists the research trace id with history", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: 10 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const traceId = "b".repeat(32);

    await useHistoryStore.getState().addFromResearch(
      "research",
      "report",
      8,
      "model",
      undefined,
      [],
      traceId,
    );

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(String(request.body)) as {
      trace_id: string;
    };
    expect(body.trace_id).toBe(traceId);
    expect(useHistoryStore.getState().entries[0].trace_id).toBe(traceId);
  });
});
