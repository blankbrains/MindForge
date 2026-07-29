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
});
