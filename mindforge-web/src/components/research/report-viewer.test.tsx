import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

import { ReportViewer } from "./report-viewer";

afterEach(cleanup);

describe("ReportViewer quality state", () => {
  it("shows not evaluated instead of a zero score", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "研究报告",
          metadata: {
            quality: null,
            quality_status: "not_evaluated",
          },
        }}
      />,
    );

    expect(screen.getByText("未评审")).not.toBeNull();
    expect(screen.queryByText("0.0 / 10")).toBeNull();
  });

  it("shows evaluation failure without a fabricated score", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "研究报告",
          metadata: {
            quality: null,
            quality_status: "evaluation_failed",
          },
        }}
      />,
    );

    expect(screen.getByText("评审失败")).not.toBeNull();
    expect(screen.queryByText("5.0 / 10")).toBeNull();
  });

  it("keeps a real critic score", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "研究报告",
          metadata: {
            quality: 8.25,
            quality_status: "evaluated",
          },
        }}
      />,
    );

    expect(screen.getByText("8.3 / 10")).not.toBeNull();
  });

  it("explains that refinement failed while keeping the valid report", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "评审前报告",
          data: {
            refinement_failure: "TimeoutError",
          },
          metadata: {
            outcome: "degraded",
            failure_reason: "报告精炼未完成",
            quality: 6.5,
            quality_status: "evaluated",
          },
        }}
      />,
    );

    expect(
      screen.getByText("报告精炼未完成，当前展示评审前的有效版本"),
    ).not.toBeNull();
    expect(screen.getByText("6.5 / 10")).not.toBeNull();
  });

  it("labels a successful model-only answer as unverified, not incomplete", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "模型知识回答",
          data: {
            grounding_status: "model_only",
          },
          metadata: {
            outcome: "success",
            grounding_status: "model_only",
            source_warning: "web_search:native_timeout",
            quality: null,
            quality_status: "not_evaluated",
          },
        }}
      />,
    );

    expect(
      screen.getByText("未获得可核验来源，当前为模型知识回答"),
    ).not.toBeNull();
    expect(screen.queryByText("部分子任务未完成，当前报告为降级结果"))
      .toBeNull();
  });

  it("explains source-required degradation without claiming task failure", () => {
    render(
      <ReportViewer
        result={{
          success: true,
          output: "模型知识回答",
          data: {
            grounding_status: "model_only",
          },
          metadata: {
            outcome: "degraded",
            grounding_status: "model_only",
            failure_reason: "联网检索未获得可核验来源",
            quality: null,
            quality_status: "not_evaluated",
          },
        }}
      />,
    );

    expect(
      screen.getByText("来源检索未完成，当前为模型知识回答"),
    ).not.toBeNull();
    expect(screen.queryByText("部分子任务未完成，当前报告为降级结果"))
      .toBeNull();
  });
});
