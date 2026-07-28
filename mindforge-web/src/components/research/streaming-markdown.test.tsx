import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StreamingAnswerPanel } from "@/components/research/streaming-markdown";
import { useResearchStore } from "@/store/research-store";

describe("StreamingAnswerPanel", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useResearchStore.getState().reset();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  it("throttles growing markdown snapshots", () => {
    render(<StreamingAnswerPanel />);

    act(() => {
      useResearchStore.setState({ streamingAnswer: "first" });
    });
    expect(screen.queryByText("first")).toBeNull();

    act(() => {
      useResearchStore.setState({ streamingAnswer: "first second" });
      vi.advanceTimersByTime(349);
    });
    expect(screen.queryByText("first second")).toBeNull();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByText("first second")).toBeTruthy();

  });
});
