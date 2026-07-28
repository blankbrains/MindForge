import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SSEEvent } from "@/types/research";
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
});
