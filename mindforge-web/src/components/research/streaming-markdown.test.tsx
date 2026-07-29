import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  StreamingAnswerPanel,
  StreamingMarkdown,
} from "@/components/research/streaming-markdown";
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

  it("adds syntax-highlighting classes to fenced code", () => {
    const { container } = render(
      <StreamingMarkdown
        content={"```python\ndef answer():\n    return 42\n```"}
      />,
    );

    const code = container.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code?.classList.contains("hljs")).toBe(true);
    expect(code?.classList.contains("language-python")).toBe(true);
  });
});
