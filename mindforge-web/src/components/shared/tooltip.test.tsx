import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Tooltip } from "@/components/shared/tooltip";

describe("Tooltip", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows supplementary guidance on hover and removes it on leave", () => {
    vi.useFakeTimers();
    render(
      <Tooltip content="由系统自动选择相关历史内容" delay={0}>
        <button type="button">自动</button>
      </Tooltip>,
    );

    const button = screen.getByRole("button", { name: "自动" });
    expect(screen.queryByRole("tooltip")).toBeNull();

    fireEvent.mouseEnter(button.parentElement!);
    act(() => {
      vi.runAllTimers();
    });
    expect(screen.getByRole("tooltip").textContent).toContain(
      "由系统自动选择相关历史内容",
    );
    expect(button.getAttribute("aria-describedby")).toBeTruthy();

    fireEvent.mouseLeave(button.parentElement!);
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("supports keyboard focus and Escape dismissal", async () => {
    render(
      <Tooltip content="切换显示模式" delay={0}>
        <button type="button">主题</button>
      </Tooltip>,
    );

    const button = screen.getByRole("button", { name: "主题" });
    fireEvent.focus(button);
    await waitFor(() => {
      expect(
        document.querySelector<HTMLElement>('[role="tooltip"]')?.style
          .visibility,
      ).toBe("visible");
    });

    fireEvent.keyDown(button.parentElement!, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("keeps only the most recently opened tooltip visible", () => {
    render(
      <>
        <Tooltip content="自动说明" delay={0}>
          <button type="button">自动</button>
        </Tooltip>
        <Tooltip content="手动说明" delay={0}>
          <button type="button">手动</button>
        </Tooltip>
      </>,
    );

    const automatic = screen.getByRole("button", { name: "自动" });
    const manual = screen.getByRole("button", { name: "手动" });
    fireEvent.mouseEnter(automatic.parentElement!);
    expect(document.querySelectorAll('[role="tooltip"]')).toHaveLength(1);

    fireEvent.focus(manual);
    const tooltips = document.querySelectorAll('[role="tooltip"]');
    expect(tooltips).toHaveLength(1);
    expect(tooltips[0]?.textContent).toBe("手动说明");
  });
});
