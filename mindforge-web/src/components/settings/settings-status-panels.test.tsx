import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ResearchSettingsPanel } from "@/components/settings/research-settings-panel";
import { RetrievalSettingsPanel } from "@/components/settings/retrieval-settings-panel";
import { useSettingsStore } from "@/store/settings-store";

beforeEach(() => {
  useSettingsStore.setState({
    researchMode: "balanced",
    tavilyConfigured: false,
    nativeWebSearchProtocol: "none",
    nativeWebSearchSupported: false,
    duckDuckGoEnabled: false,
    modelOnlyFallbackEnabled: true,
    webSearchAvailable: false,
    queueTimeout: 30,
    nativeWebSearchTimeoutSeconds: 30,
    sandboxTimeout: 15,
    rerankerConfigured: true,
    rerankerAvailable: false,
    rerankerLoadFailed: true,
  });
});

afterEach(cleanup);

describe("settings status panels", () => {
  it("gives the selected research mode a strong, accessible state", () => {
    render(<ResearchSettingsPanel />);

    const selected = screen.getByRole("button", { name: /^均衡简单/ });
    const unselected = screen.getByRole("button", { name: /^快速单任务/ });

    expect(selected.getAttribute("aria-pressed")).toBe("true");
    expect(selected.className).toContain("bg-primary/15");
    expect(selected.className).toContain("ring-2");
    expect(unselected.getAttribute("aria-pressed")).toBe("false");
  });

  it("explains unavailable reranking and model-only search fallback", () => {
    render(<RetrievalSettingsPanel />);

    expect(screen.getByText("模型加载失败")).toBeTruthy();
    expect(
      screen.getByText("继续使用基础混合检索；请检查模型文件、依赖和设备配置"),
    ).toBeTruthy();
    expect(screen.getByText("当前无联网后端")).toBeTruthy();
    expect(
      screen.getByText(
        "研究会保留模型回答，并明确标记为无可核验引用",
      ),
    ).toBeTruthy();
    expect(
      screen.getByRole("spinbutton", {
        name: "原生联网搜索超时（秒）",
      }),
    ).toHaveProperty("value", "30");
  });

  it("shows runtime tool timeout controls", () => {
    render(<ResearchSettingsPanel />);

    expect(
      screen.getByRole("spinbutton", {
        name: "工具排队超时（秒）",
      }),
    ).toHaveProperty("value", "30");
    expect(
      screen.getByRole("spinbutton", {
        name: "代码执行超时（秒）",
      }),
    ).toHaveProperty("value", "15");
  });
});
