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

  it("wraps GFM tables for bordered responsive rendering", () => {
    const { container } = render(
      <StreamingMarkdown
        content={
          "| 模型 | 费用 |\n| --- | ---: |\n| DeepSeek | $0.01 |"
        }
      />,
    );

    const wrapper = container.querySelector(".markdown-table-scroll");
    expect(wrapper).not.toBeNull();
    expect(wrapper?.querySelector("table")).not.toBeNull();
    expect(wrapper?.querySelectorAll("th")).toHaveLength(2);
    expect(wrapper?.querySelectorAll("td")).toHaveLength(2);
  });

  it("links citation markers to safe external sources", () => {
    render(
      <StreamingMarkdown
        content="This claim is supported [1]."
        sources={[
          {
            index: 1,
            title: "Example source",
            url: "https://example.com/article",
            source: "web",
          },
        ]}
      />,
    );

    const citation = screen.getByRole("link", {
      name: "查看来源 1",
    });
    expect(citation.getAttribute("href")).toBe(
      "https://example.com/article",
    );
    expect(citation.getAttribute("target")).toBe("_blank");
    expect(citation.getAttribute("rel")).toBe("noreferrer noopener");
    expect(
      screen.getByRole("heading", { name: "参考文献" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Example source" }),
    ).toBeTruthy();
  });

  it("links internal sources to their rendered source entry", () => {
    const { container } = render(
      <StreamingMarkdown
        content="Internal evidence [2]."
        sources={[
          {
            index: 2,
            title: "Knowledge document",
            url: "",
            source: "knowledge_base",
          },
        ]}
      />,
    );

    const citation = screen.getByRole("link", {
      name: "查看来源 2",
    });
    expect(citation.getAttribute("href")).toBe("#research-source-2");
    expect(citation.getAttribute("target")).toBeNull();
    expect(container.querySelector("#research-source-2")).not.toBeNull();
  });

  it("does not rewrite citations inside code or existing links", () => {
    const { container } = render(
      <StreamingMarkdown
        content={
          "`[1]`\n\n```text\n[1]\n```\n\n[[1]](https://existing.example/path)"
        }
        sources={[
          {
            index: 1,
            title: "Citation source",
            url: "https://citation.example/source",
            source: "web",
          },
        ]}
      />,
    );

    expect(container.querySelectorAll("a.citation-link")).toHaveLength(0);
    const existing = screen.getByRole("link", { name: "[1]" });
    expect(existing.getAttribute("href")).toBe(
      "https://existing.example/path",
    );
    expect(container.querySelector("code")?.textContent).toBe("[1]");
    expect(container.querySelector("pre code")?.textContent).toBe("[1]\n");
  });

  it("keeps unknown markers as text and rejects unsafe source URLs", () => {
    const { container } = render(
      <StreamingMarkdown
        content="Unknown [99], unsafe [1]."
        sources={[
          {
            index: 1,
            title: "Unsafe source",
            url: "javascript:alert(1)",
            source: "web",
          },
        ]}
      />,
    );

    expect(container.textContent).toContain("Unknown [99]");
    const citation = container.querySelector(
      'a[aria-label="查看来源 1"]',
    );
    expect(citation).not.toBeNull();
    expect(citation?.getAttribute("href")).toBe("#research-source-1");
    expect(
      container.querySelector('a[href^="javascript:"]'),
    ).toBeNull();
    expect(container.querySelector("#research-source-1 a")).toBeNull();
  });
});
