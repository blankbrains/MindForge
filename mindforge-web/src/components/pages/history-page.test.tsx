import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const historyState = vi.hoisted(() => {
  let resolveFirst!: () => void;
  let resolveSecond!: () => void;
  return {
    state: {
      entries: [
        {
          id: 1,
          task: "first task",
          report: "first report",
          quality_score: 8,
          model_used: "test",
          created_at: "2026-07-27T00:00:00Z",
        },
        {
          id: 2,
          task: "second task",
          report: "second report",
          quality_score: 8,
          model_used: "test",
          created_at: "2026-07-27T00:00:00Z",
        },
      ],
      loaded: true,
      loadHistory: vi.fn(async () => {}),
      loadEntry: vi.fn(
        (id: number) =>
          new Promise<void>((resolve) => {
            if (id === 1) resolveFirst = resolve;
            else resolveSecond = resolve;
          }),
      ),
      removeEntry: vi.fn(async () => {}),
      clearAll: vi.fn(async () => {}),
    },
    resolveFirst: () => resolveFirst(),
    resolveSecond: () => resolveSecond(),
  };
});

vi.mock("@/store/history-store", () => ({
  useHistoryStore: (
    selector?: (state: typeof historyState.state) => unknown,
  ) => (selector ? selector(historyState.state) : historyState.state),
}));

import { HistoryPage } from "./history-page";

describe("HistoryPage", () => {
  it("does not let an older detail request replace the latest selection", async () => {
    render(<HistoryPage />);
    fireEvent.click(screen.getByText("first task"));
    fireEvent.click(screen.getByText("second task"));

    historyState.resolveSecond();
    expect(await screen.findByText("second report")).not.toBeNull();

    historyState.resolveFirst();
    await waitFor(() => {
      expect(screen.queryByText("first report")).toBeNull();
      expect(screen.getByText("second report")).not.toBeNull();
    });
  });
});
