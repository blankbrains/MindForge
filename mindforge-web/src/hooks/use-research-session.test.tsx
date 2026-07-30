import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SSEEvent } from "@/types/research";
import { useHistoryStore } from "@/store/history-store";
import { useResearchStore } from "@/store/research-store";
import { useSettingsStore } from "@/store/settings-store";

interface CapturedConnection {
  onEvent: (event: SSEEvent) => void;
  onComplete: () => void;
  onError: (error: Error) => void;
  abort: ReturnType<typeof vi.fn>;
}

const sseState = vi.hoisted(() => ({
  connections: [] as CapturedConnection[],
}));

vi.mock("@/lib/sse-parser", () => ({
  createSSEConnection: vi.fn(
    (
      _url: string,
      _body: unknown,
      onEvent: (event: SSEEvent) => void,
      onComplete: () => void,
      onError: (error: Error) => void,
    ) => {
      const connection = {
        onEvent,
        onComplete,
        onError,
        abort: vi.fn(),
      };
      sseState.connections.push(connection);
      return { abort: connection.abort };
    },
  ),
}));

import { useResearchSession } from "@/hooks/use-research-session";

describe("useResearchSession", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sseState.connections.length = 0;
    useResearchStore.getState().reset();
    useHistoryStore.setState({
      addFromResearch: vi.fn().mockResolvedValue(undefined),
    });
    useSettingsStore.setState({ researchTimeout: 180 });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it("ignores callbacks from a superseded SSE request", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("first"));
    act(() => result.current.startResearch("second"));

    expect(sseState.connections).toHaveLength(2);
    expect(sseState.connections[0].abort).toHaveBeenCalledOnce();

    act(() => {
      sseState.connections[0].onEvent({
        type: "answer_chunk",
        content: "stale",
      });
      sseState.connections[0].onError(new Error("stale failure"));
      sseState.connections[0].onComplete();
    });

    expect(useResearchStore.getState().task).toBe("second");
    expect(useResearchStore.getState().streamingAnswer).toBe("");
    expect(useResearchStore.getState().status).toBe("streaming");

    unmount();
  });

  it("tracks planning and ignores heartbeat state changes", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("planning test"));
    act(() => {
      sseState.connections[0].onEvent({
        type: "planning",
        status: "start",
      });
    });

    expect(useResearchStore.getState().planning).toBe(true);

    act(() => {
      sseState.connections[0].onEvent({
        type: "heartbeat",
        timestamp: Date.now(),
      });
    });

    expect(useResearchStore.getState().planning).toBe(true);
    expect(useResearchStore.getState().status).toBe("streaming");

    act(() => {
      sseState.connections[0].onEvent({
        type: "planning",
        status: "done",
      });
    });

    expect(useResearchStore.getState().planning).toBe(false);
    unmount();
  });

  it("names the selected provider in authentication errors", () => {
    useSettingsStore.setState({
      llmProvider: "local",
      providerConfigs: {
        ...useSettingsStore.getState().providerConfigs,
        local: {
          ...useSettingsStore.getState().providerConfigs.local,
          label: "本地模型",
        },
      },
    });
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("provider error"));
    const error = Object.assign(new Error("unauthorized"), { status: 401 });
    act(() => sseState.connections[0].onError(error));

    expect(useResearchStore.getState().error).toContain("本地模型");
    expect(useResearchStore.getState().error).not.toContain("DeepSeek");
    unmount();
  });

  it("clears the submitted question only after a successful result", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("completed question"));
    act(() => {
      sseState.connections[0].onEvent({
        type: "done",
        result: {
          success: true,
          output: "report",
        },
      });
    });

    expect(useResearchStore.getState().task).toBe("");
    expect(useResearchStore.getState().status).toBe("completed");
    unmount();
  });

  it("keeps the submitted question when research fails", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("retry this question"));
    act(() => {
      sseState.connections[0].onEvent({
        type: "done",
        result: {
          success: false,
          output: "failed",
        },
      });
    });

    expect(useResearchStore.getState().task).toBe("retry this question");
    expect(useResearchStore.getState().status).toBe("error");
    unmount();
  });

  it("persists structured sources with a successful report", () => {
    const addFromResearch = vi.fn().mockResolvedValue(undefined);
    useHistoryStore.setState({ addFromResearch });
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("citation test"));
    act(() => {
      sseState.connections[0].onEvent({
        type: "done",
        result: {
          success: true,
          output: "Report [1]",
          data: {
            sources: [
              {
                index: 1,
                title: "Source",
                url: "https://example.com/source",
                source: "web",
                content: "must not be persisted",
              },
            ],
          },
        },
      });
    });

    expect(addFromResearch).toHaveBeenCalledWith(
      "citation test",
      "Report [1]",
      undefined,
      undefined,
      {
        tokenUsage: undefined,
        costUsd: undefined,
        costStatus: undefined,
      },
      [
        {
          index: 1,
          title: "Source",
          url: "https://example.com/source",
          source: "web",
        },
      ],
    );
    unmount();
  });
});
