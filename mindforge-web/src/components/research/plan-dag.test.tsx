import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PlanDAG } from "@/components/research/plan-dag";
import type { ResearchPlan } from "@/types/research";

vi.mock("@xyflow/react", async () => {
  const React = await import("react");
  return {
    Background: () => null,
    Controls: () => null,
    ReactFlow: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="react-flow">{children}</div>
    ),
    useEdgesState: <T,>(initial: T[]) => {
      const [edges, setEdges] = React.useState(initial);
      return [edges, setEdges, vi.fn()] as const;
    },
    useNodesState: <T,>(initial: T[]) => {
      const [nodes, setNodes] = React.useState(initial);
      return [nodes, setNodes, vi.fn()] as const;
    },
  };
});

function plan(
  plannerStatus: ResearchPlan["planner_status"],
  plannerError: string | null = null,
): ResearchPlan {
  return {
    plan_id: "plan",
    original_task: "比较 Python 和 Java",
    reasoning: "",
    planner_status: plannerStatus,
    planner_error: plannerError,
    subtasks: [
      {
        task_id: "t1",
        description: "研究问题",
        task_type: "research",
        dependencies: [],
        status: "pending",
        priority: 1,
      },
    ],
  };
}

afterEach(cleanup);

describe("PlanDAG planner status", () => {
  it("explains an intentional direct single-task plan", () => {
    render(<PlanDAG plan={plan("direct")} />);

    expect(
      screen.getByText("均衡模式判定该问题范围集中，直接使用单个研究任务。"),
    ).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("surfaces planner fallback details", () => {
    render(<PlanDAG plan={plan("fallback", "invalid JSON")} />);

    expect(screen.getByRole("alert").textContent).toContain(
      "Planner 规划失败，已降级为单任务：invalid JSON",
    );
  });
});
