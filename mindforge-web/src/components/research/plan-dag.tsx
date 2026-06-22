import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ResearchPlan } from "@/types/research";
import { cn } from "@/lib/utils";

interface Props {
  plan: ResearchPlan | null;
}

export function PlanDAG({ plan }: Props) {
  const { nodes, edges } = useMemo(() => {
    if (!plan || plan.subtasks.length === 0) {
      return { nodes: [], edges: [] };
    }

    const flowNodes: Node[] = plan.subtasks.map((st, i) => ({
      id: st.task_id,
      type: "default",
      position: { x: 0, y: i * 120 },
      data: {
        label: (
          <div
            className={cn(
              "rounded-lg border-2 px-4 py-3 text-sm font-medium shadow-sm min-w-[200px]",
              st.status === "completed" &&
                "border-green-400 bg-green-50 text-green-800 dark:border-green-600 dark:bg-green-950 dark:text-green-200",
              st.status === "in_progress" &&
                "border-blue-400 bg-blue-50 text-blue-800 dark:border-blue-600 dark:bg-blue-950 dark:text-blue-200",
              st.status === "failed" &&
                "border-red-400 bg-red-50 text-red-800 dark:border-red-600 dark:bg-red-950 dark:text-red-200",
              st.status === "pending" &&
                "border-border bg-surface text-text dark:border-border dark:bg-surface",
            )}
          >
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  st.status === "completed" && "bg-green-500",
                  st.status === "in_progress" && "bg-blue-500 animate-pulse",
                  st.status === "failed" && "bg-red-500",
                  st.status === "pending" && "bg-border",
                )}
              />
              <span className="text-[10px] uppercase tracking-wider opacity-60">
                {st.task_type}
              </span>
            </div>
            <p className="mt-1.5 text-xs leading-relaxed">{st.description}</p>
          </div>
        ),
      },
    }));

    const flowEdges: Edge[] = plan.subtasks.flatMap((st) =>
      (st.dependencies ?? []).map((dep) => ({
        id: `${dep}->${st.task_id}`,
        source: dep,
        target: st.task_id,
        animated: st.status === "in_progress",
        style: { stroke: "#a29bfe", strokeWidth: 2 },
      })),
    );

    return { nodes: flowNodes, edges: flowEdges };
  }, [plan]);

  if (!plan) return null;

  return (
    <div className="h-[400px] w-full rounded-xl border border-border bg-surface">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <Controls />
        <MiniMap
          nodeColor={() => "#a29bfe"}
        />
      </ReactFlow>
    </div>
  );
}
