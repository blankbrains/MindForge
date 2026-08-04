import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  refetch: vi.fn(),
  deleteTrace: vi.fn(),
  clearTraces: vi.fn(),
  traceStatus: "degraded",
  failureCount: 1,
}));

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-router")>();
  return {
    ...actual,
    useNavigate: () => mocks.navigate,
    useSearch: () => ({ traceId: "a".repeat(32) }),
  };
});

vi.mock("@/hooks/use-observability", () => ({
  useDeleteTrace: () => ({
    mutateAsync: mocks.deleteTrace,
    isPending: false,
  }),
  useClearTraces: () => ({
    mutateAsync: mocks.clearTraces,
    isPending: false,
  }),
  useObservabilityStatus: () => ({
    data: {
      enabled: true,
      local_storage: true,
      remote_configured: true,
      langfuse_host: "https://langfuse.example",
      capture_content: false,
      retention_days: 7,
    },
    isError: false,
    isFetching: false,
    refetch: mocks.refetch,
  }),
  useTraceList: () => ({
    data: {
      traces: [
        {
          trace_id: "a".repeat(32),
          name: "orchestrator.research",
          display_name: "Python vs Java",
          start_time: 1_800_000_000,
          end_time: 1_800_000_002,
          duration_ms: 2000,
          status: mocks.traceStatus,
          error: null,
          failure_summary: null,
          failure_count: mocks.failureCount,
          task_preview: null,
          metadata: {},
          span_count: 3,
          generation_count: 1,
          tool_count: 1,
          error_count: 0,
          total_tokens: 42,
          cost_usd: 0.001,
          cost_status: "estimated",
          remote_url: "https://langfuse.example/trace/test",
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      truncated: false,
    },
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: mocks.refetch,
  }),
  useTraceDetail: () => ({
    data: {
      summary: {
        trace_id: "a".repeat(32),
        name: "orchestrator.research",
        display_name: "Python vs Java",
        start_time: 1_800_000_000,
        end_time: 1_800_000_002,
        duration_ms: 2000,
        status: mocks.traceStatus,
        error: "LLM request timed out after 45 seconds.",
        failure_summary:
          "检测到 1 个异常节点。主要原因：LLM 请求在 45 秒后超时。",
        failure_count: mocks.failureCount,
        task_preview: null,
        metadata: {},
        span_count: 3,
        generation_count: 1,
        tool_count: 1,
        error_count: 0,
        total_tokens: 42,
        cost_usd: 0.001,
        cost_status: "estimated",
        remote_url: "https://langfuse.example/trace/test",
      },
      observations: [
        {
          span_id: "1".repeat(16),
          trace_id: "a".repeat(32),
          name: "orchestrator.research",
          start_time: 1_800_000_000,
          end_time: 1_800_000_002,
          duration_ms: 2000,
          parent_id: null,
          error: null,
          metadata: {},
        },
        {
          span_id: "2".repeat(16),
          trace_id: "a".repeat(32),
          name: "llm.chat",
          start_time: 1_800_000_000.2,
          end_time: 1_800_000_001,
          duration_ms: 800,
          parent_id: "1".repeat(16),
          error: "LLM request timed out after 45 seconds.",
          metadata: {
            agent: "researcher",
            model: "test-model",
            attempt: 1,
            stage: "llm_request",
            error_code: "llm_request_timeout",
            error_type: "TimeoutError",
          },
        },
        {
          span_id: "3".repeat(16),
          trace_id: "a".repeat(32),
          name: "tool.execute",
          start_time: 1_800_000_001,
          end_time: 1_800_000_001.5,
          duration_ms: 500,
          parent_id: "1".repeat(16),
          error: null,
          metadata: { tool: "web_search" },
        },
      ],
      failures: [
        {
          span_id: "2".repeat(16),
          parent_id: "1".repeat(16),
          observation_name: "llm.chat",
          stage: "llm_request",
          error_code: "llm_request_timeout",
          error_type: "TimeoutError",
          message: "LLM 请求在 45 秒后超时。",
          status: "error",
          agent: "researcher",
          model: "test-model",
          attempt: 1,
        },
      ],
      observations_truncated: false,
    },
    isLoading: false,
    isError: false,
    refetch: mocks.refetch,
  }),
}));

import { ObservabilityPage } from "./observability-page";

describe("ObservabilityPage", () => {
  beforeEach(() => {
    mocks.navigate.mockReset();
    mocks.refetch.mockReset();
    mocks.deleteTrace.mockReset();
    mocks.clearTraces.mockReset();
    mocks.traceStatus = "degraded";
    mocks.failureCount = 1;
  });

  it("shows backend status and the complete trace chain", () => {
    render(<ObservabilityPage />);

    expect(screen.getByText("可观测")).toBeTruthy();
    expect(screen.getByText("已配置")).toBeTruthy();
    expect(screen.getAllByText("Python vs Java").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Orchestrator").length).toBeGreaterThan(0);
    expect(screen.getByText("LLM 调用")).toBeTruthy();
    expect(screen.getAllByText("工具调用").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/检测到 1 个异常节点。主要原因/),
    ).toBeTruthy();
    expect(screen.getByText("llm_request_timeout")).toBeTruthy();
    expect(screen.getByText("test-model")).toBeTruthy();
    expect(screen.getByText("第 1 次")).toBeTruthy();
    const langfuseLink = screen.getByRole("link", {
      name: /在 Langfuse 中打开/,
    });
    expect(langfuseLink.getAttribute("href")).toBe(
      "https://langfuse.example/trace/test",
    );
  });

  it("shows recovered child failures as success with warnings", () => {
    mocks.traceStatus = "success";
    mocks.failureCount = 2;

    render(<ObservabilityPage />);

    expect(screen.getAllByText("成功（有告警）").length).toBeGreaterThan(0);
    expect(screen.getByText(/链路异常汇总（最终已恢复）/)).toBeTruthy();
    expect(
      screen.getAllByRole("option", { name: "成功（有告警）" }).length,
    ).toBeGreaterThan(0);
  });
});
