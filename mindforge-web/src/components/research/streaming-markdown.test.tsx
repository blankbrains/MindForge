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

  it("preserves an explicit language instead of auto-detecting it", () => {
    const { container } = render(
      <StreamingMarkdown
        content={
          "```java\npublic class Main {\n  public static void main(String[] args) {}\n}\n```"
        }
      />,
    );

    const code = container.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code?.classList.contains("hljs")).toBe(true);
    expect(code?.classList.contains("language-java")).toBe(true);
    expect(code?.classList.contains("language-csharp")).toBe(false);
  });

  it("auto-detects the language of an unlabeled code block", () => {
    const { container } = render(
      <StreamingMarkdown
        content={
          "```\ndef greet(name):\n    message = f\"Hello, {name}\"\n    return message\n```"
        }
      />,
    );

    const code = container.querySelector("pre code");
    expect(code?.classList.contains("hljs")).toBe(true);
    expect(code?.classList.contains("language-python")).toBe(true);
  });

  it("keeps explicit text blocks unhighlighted", () => {
    const { container } = render(
      <StreamingMarkdown content={"```text\nnot code: just a note\n```"} />,
    );

    const code = container.querySelector("pre code");
    expect(code?.classList.contains("language-text")).toBe(true);
    expect(code?.classList.contains("hljs")).toBe(false);
    expect(code?.querySelector("[class^='hljs-']")).toBeNull();
  });

  it("supports explicitly labeled PowerShell blocks", () => {
    const { container } = render(
      <StreamingMarkdown
        content={'```powershell\nGet-ChildItem | Where-Object { $_.Length -gt 0 }\n```'}
      />,
    );

    const code = container.querySelector("pre code");
    expect(code?.classList.contains("hljs")).toBe(true);
    expect(code?.classList.contains("language-powershell")).toBe(true);
    expect(code?.querySelector(".hljs-built_in")).not.toBeNull();
  });
});
