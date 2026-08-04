import { useState } from "react";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { QueryInput } from "@/components/research/query-input";

afterEach(cleanup);

function RunningQueryInput({
  onSubmit,
  onCancel,
}: {
  onSubmit: (task: string) => void;
  onCancel: () => void;
}) {
  const [isRunning, setIsRunning] = useState(true);

  return (
    <QueryInput
      value="不能被重新提交"
      onChange={vi.fn()}
      onSubmit={onSubmit}
      isRunning={isRunning}
      onCancel={() => {
        onCancel();
        setIsRunning(false);
      }}
    />
  );
}

describe("QueryInput", () => {
  it("labels retrieval-only submissions accurately", () => {
    render(
      <QueryInput
        value="MindForge"
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        isRunning={false}
        onCancel={vi.fn()}
        retrievalOnly
      />,
    );

    expect(
      screen.getByRole("button", { name: "知识库检索" }),
    ).toBeTruthy();
  });

  it("submits the trimmed research task", () => {
    const onSubmit = vi.fn();
    render(
      <QueryInput
        value="  MindForge  "
        onChange={vi.fn()}
        onSubmit={onSubmit}
        isRunning={false}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));

    expect(onSubmit).toHaveBeenCalledWith("MindForge");
  });

  it("locks the editor and still cancels an active research task", () => {
    const onCancel = vi.fn();
    const onChange = vi.fn();
    render(
      <QueryInput
        value="下一条问题"
        onChange={onChange}
        onSubmit={vi.fn()}
        isRunning
        onCancel={onCancel}
      />,
    );

    const input = screen.getByRole("textbox");
    expect(input).toHaveProperty("readOnly", true);
    expect(input).not.toHaveProperty("disabled", true);
    fireEvent.change(input, { target: { value: "新的草稿" } });
    fireEvent.click(screen.getByRole("button", { name: "停止研究" }));

    expect(onChange).not.toHaveBeenCalled();
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("does not submit the form when cancellation synchronously ends the run", () => {
    const onSubmit = vi.fn();
    const onCancel = vi.fn();
    render(<RunningQueryInput onSubmit={onSubmit} onCancel={onCancel} />);

    const stopButton = screen.getByRole("button", { name: "停止研究" });
    fireEvent.click(stopButton);

    expect(onCancel).toHaveBeenCalledOnce();
    expect(onSubmit).not.toHaveBeenCalled();
    expect(stopButton.getAttribute("type")).toBe("button");
    expect(screen.getByRole("button", { name: "开始研究" })).toBeTruthy();
    expect(screen.getByRole("textbox")).toHaveProperty("readOnly", false);
  });
});
