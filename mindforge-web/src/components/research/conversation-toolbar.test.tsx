import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConversationToolbar } from "@/components/research/conversation-toolbar";

describe("ConversationToolbar", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("explains context modes and keeps mode selection interactive", () => {
    vi.useFakeTimers();
    const onModeChange = vi.fn();
    render(
      <ConversationToolbar
        conversations={[
          {
            conversation_id: "a".repeat(32),
            title: "当前会话",
            status: "active",
            context_mode: "auto",
            version: 1,
            created_at: "2026-08-05T00:00:00Z",
            updated_at: "2026-08-05T00:00:00Z",
          },
        ]}
        activeConversationId={"a".repeat(32)}
        contextMode="auto"
        independent={false}
        disabled={false}
        loading={false}
        onSelect={vi.fn()}
        onCreate={vi.fn()}
        onDelete={vi.fn()}
        onModeChange={onModeChange}
        onIndependentChange={vi.fn()}
        onOpenContext={vi.fn()}
      />,
    );

    const automatic = screen.getByRole("button", { name: "自动" });
    fireEvent.mouseEnter(automatic.parentElement!);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(document.querySelector('[role="tooltip"]')?.textContent).toContain(
      "自动选择上下文",
    );

    fireEvent.click(screen.getByRole("button", { name: "手动" }));
    expect(onModeChange).toHaveBeenCalledWith("manual");
  });
});
