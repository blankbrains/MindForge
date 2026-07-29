import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { QueryInput } from "@/components/research/query-input";

describe("QueryInput", () => {
  it("labels retrieval-only submissions accurately", () => {
    render(
      <QueryInput
        value="MindForge"
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        disabled={false}
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
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "开始研究" }));

    expect(onSubmit).toHaveBeenCalledWith("MindForge");
  });
});
