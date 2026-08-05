import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SSEEvent } from "@/types/research";
import { useHistoryStore } from "@/store/history-store";
import { useResearchStore } from "@/store/research-store";
import { useSettingsStore } from "@/store/settings-store";

interface CapturedConnection {
  body: Record<string, unknown>;
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
      body: Record<string, unknown>,
      onEvent: (event: SSEEvent) => void,
      onComplete: () => void,
      onError: (error: Error) => void,
    ) => {
      const connection = {
        body,
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

import {
  cancelResearch,
  useResearchSession,
} from "@/hooks/use-research-session";

async function flushCancellationRequest(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("useResearchSession", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    sseState.connections.length = 0;
    useResearchStore.getState().reset();
    useHistoryStore.setState({
      addFromResearch: vi.fn().mockResolvedValue(undefined),
    });
    useSettingsStore.setState({ researchTimeout: 180 });
  });

  afterEach(async () => {
    act(() => cancelResearch());
    await flushCancellationRequest();
    useResearchStore.getState().reset();
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("ignores callbacks from a superseded SSE request", async () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("first"));
    act(() => result.current.startResearch("second"));
    await flushCancellationRequest();

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

  it("fully clears the active run when the user cancels", async () => {
    const { result, unmount } = renderHook(() => useResearchSession());
    let finishCancellation!: () => void;
    vi.mocked(fetch).mockReturnValueOnce(
      new Promise((resolve) => {
        finishCancellation = () => resolve({ ok: true } as Response);
      }),
    );

    act(() => result.current.startResearch("cancel me"));
    act(() => {
      sseState.connections[0].onEvent({
        type: "planning",
        status: "start",
      });
      result.current.cancelResearch();
    });

    const state = useResearchStore.getState();
    expect(sseState.connections[0].abort).not.toHaveBeenCalled();
    expect(state.status).toBe("cancelled");
    expect(state.task).toBe("cancel me");
    expect(state.activeTask).toBe("");
    expect(state.startedAt).toBeNull();
    expect(state.planning).toBe(false);
    expect(state.phase).toBe("cancelled");
    const requestId = sseState.connections[0].body.request_id;
    expect(requestId).toEqual(expect.any(String));
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/query/cancel",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ request_id: requestId }),
        keepalive: true,
      }),
    );
    finishCancellation();
    await flushCancellationRequest();
    expect(sseState.connections[0].abort).toHaveBeenCalledOnce();

    act(() => {
      sseState.connections[0].onEvent({
        type: "answer_chunk",
        content: "late response",
      });
      sseState.connections[0].onComplete();
    });
    expect(useResearchStore.getState().status).toBe("cancelled");
    expect(useResearchStore.getState().streamingAnswer).toBe("");

    unmount();
  });

  it("keeps the active run across page unmount and remount", async () => {
    const first = renderHook(() => useResearchSession());

    act(() => first.result.current.startResearch("navigate away"));
    expect(useResearchStore.getState().status).toBe("streaming");

    first.unmount();
    await flushCancellationRequest();

    expect(sseState.connections[0].abort).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
    expect(useResearchStore.getState().status).toBe("streaming");
    expect(useResearchStore.getState().activeTask).toBe("navigate away");

    const second = renderHook(() => useResearchSession());
    expect(second.result.current.isStreaming).toBe(true);
    expect(second.result.current.startedAt).not.toBeNull();

    act(() => second.result.current.cancelResearch());
    await flushCancellationRequest();
    expect(sseState.connections[0].abort).toHaveBeenCalledOnce();
    expect(useResearchStore.getState().status).toBe("cancelled");

    second.unmount();
  });

  it("repairs a stale running state when the page mounts again", () => {
    useResearchStore.setState({
      status: "streaming",
      task: "stale question",
      activeTask: "stale question",
      phase: "researching",
      startedAt: Date.now() - 24 * 60 * 1000,
    });

    const { unmount } = renderHook(() => useResearchSession());

    const state = useResearchStore.getState();
    expect(state.status).toBe("idle");
    expect(state.task).toBe("stale question");
    expect(state.activeTask).toBe("");
    expect(state.startedAt).toBeNull();
    expect(state.phase).toBe("idle");

    unmount();
  });

  it("cancels the active transport when the browser unloads", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("refresh me"));
    const requestId = sseState.connections[0].body.request_id;
    act(() => window.dispatchEvent(new Event("beforeunload")));

    expect(sseState.connections[0].abort).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/query/cancel",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ request_id: requestId }),
        keepalive: true,
      }),
    );

    unmount();
  });

  it("ends the run and aborts the request at the configured timeout", async () => {
    useSettingsStore.setState({ researchTimeout: 30 });
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("timeout test"));
    act(() => vi.advanceTimersByTime(30_000));
    await flushCancellationRequest();

    const state = useResearchStore.getState();
    expect(sseState.connections[0].abort).toHaveBeenCalledOnce();
    expect(state.status).toBe("error");
    expect(state.error).toContain("研究超时");
    expect(state.startedAt).toBeNull();
    expect(state.activeTask).toBe("");

    unmount();
  });

  it("clears timing state when the stream ends without a result", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() => result.current.startResearch("incomplete stream"));
    act(() => sseState.connections[0].onComplete());

    const state = useResearchStore.getState();
    expect(state.status).toBe("error");
    expect(state.error).toContain("未收到完整结果");
    expect(state.startedAt).toBeNull();
    expect(state.activeTask).toBe("");

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

  it("sends observable context controls with a conversation request", () => {
    const { result, unmount } = renderHook(() => useResearchSession());

    act(() =>
      result.current.startResearch("follow-up", {
        conversationId: "a".repeat(32),
        contextMode: "manual",
        selectedContextIds: ["message:m1"],
        excludedContextIds: ["artifact:a1"],
        independent: false,
      }),
    );

    expect(sseState.connections[0].body).toEqual(
      expect.objectContaining({
        task: "follow-up",
        conversation_id: "a".repeat(32),
        context_mode: "manual",
        selected_context_ids: ["message:m1"],
        excluded_context_ids: ["artifact:a1"],
        independent: false,
      }),
    );
    unmount();
  });
});
