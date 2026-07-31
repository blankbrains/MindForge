import {
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
  type ReactFlowInstance,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { Info, RotateCcw, TriangleAlert } from "lucide-react";
import "@xyflow/react/dist/style.css";
import type { ResearchPlan, SubTask } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { cn } from "@/lib/utils";

interface Props {
  plan: ResearchPlan | null;
}

type TaskNode = Node<{ label: ReactNode }>;

function taskDepths(subtasks: SubTask[]): Map<string, number> {
  const byId = new Map(subtasks.map((task) => [task.task_id, task]));
  const cache = new Map<string, number>();
  const visiting = new Set<string>();

  const depthOf = (taskId: string): number => {
    const cached = cache.get(taskId);
    if (cached !== undefined) return cached;
    if (visiting.has(taskId)) return 0;
    visiting.add(taskId);
    const task = byId.get(taskId);
    const dependencies = task?.dependencies ?? [];
    const depth = dependencies.length
      ? Math.max(...dependencies.map((dependency) => depthOf(dependency))) + 1
      : 0;
    visiting.delete(taskId);
    cache.set(taskId, depth);
    return depth;
  };

  for (const task of subtasks) depthOf(task.task_id);
  return cache;
}

function taskLabel(
  task: SubTask,
  index: number,
  status: SubTask["status"],
): ReactNode {
  const dependencies = task.dependencies ?? [];
  return (
    <div
      className={cn(
        "w-[260px] border-2 bg-surface px-4 py-3 text-left shadow-sm",
        status === "completed"
          && "border-green-400 bg-green-50 text-green-900 dark:border-green-700 dark:bg-green-950 dark:text-green-100",
        status === "in_progress"
          && "border-blue-400 bg-blue-50 text-blue-900 dark:border-blue-700 dark:bg-blue-950 dark:text-blue-100",
        status === "failed"
          && "border-red-400 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-950 dark:text-red-100",
        status === "pending"
          && "border-border bg-surface text-text",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold">子问题 {index + 1}</span>
        <span className="font-mono text-[10px] text-text-muted">
          {task.task_id}
        </span>
      </div>
      <p className="mt-2 text-sm font-medium leading-6">
        {task.description}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
        <span>{task.task_type}</span>
        <span>优先级 {task.priority}</span>
        {dependencies.length > 0 && (
          <span>依赖 {dependencies.join("、")}</span>
        )}
      </div>
    </div>
  );
}

function buildNodes(
  plan: ResearchPlan,
  states: Record<string, SubTask>,
): TaskNode[] {
  const depths = taskDepths(plan.subtasks);
  const rowsByDepth = new Map<number, number>();

  return plan.subtasks.map((task, index) => {
    const depth = depths.get(task.task_id) ?? 0;
    const row = rowsByDepth.get(depth) ?? 0;
    rowsByDepth.set(depth, row + 1);
    const status = states[task.task_id]?.status ?? task.status ?? "pending";
    return {
      id: task.task_id,
      position: { x: depth * 330, y: row * 190 },
      data: { label: taskLabel(task, index, status) },
      style: {
        width: 260,
        padding: 0,
        border: "none",
        background: "transparent",
      },
    };
  });
}

function buildEdges(plan: ResearchPlan): Edge[] {
  return plan.subtasks.flatMap((task) =>
    (task.dependencies ?? []).map((dependency) => ({
      id: `${dependency}->${task.task_id}`,
      source: dependency,
      target: task.task_id,
      type: "smoothstep",
      style: { stroke: "#7c83a1", strokeWidth: 1.5 },
    })),
  );
}

export function PlanDAG({ plan }: Props) {
  const subtaskStates = useResearchStore((state) => state.subtasks);
  const [nodes, setNodes, onNodesChange] = useNodesState<TaskNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const flowRef = useRef<ReactFlowInstance<TaskNode, Edge> | null>(null);

  const resetLayout = useCallback(() => {
    if (!plan) return;
    setNodes(buildNodes(plan, subtaskStates));
    setEdges(buildEdges(plan));
    window.setTimeout(() => {
      void flowRef.current?.fitView({ padding: 0.2, duration: 250 });
    }, 0);
  }, [plan, setEdges, setNodes, subtaskStates]);

  useEffect(() => {
    if (!plan) return;
    setNodes(buildNodes(plan, {}));
    setEdges(buildEdges(plan));
    window.setTimeout(() => {
      void flowRef.current?.fitView({ padding: 0.2 });
    }, 0);
  }, [plan, setEdges, setNodes]);

  useEffect(() => {
    if (!plan) return;
    setNodes((current) =>
      current.map((node) => {
        const index = plan.subtasks.findIndex(
          (task) => task.task_id === node.id,
        );
        const task = plan.subtasks[index];
        if (!task) return node;
        const status =
          subtaskStates[task.task_id]?.status ?? task.status ?? "pending";
        return {
          ...node,
          data: { label: taskLabel(task, index, status) },
        };
      }),
    );
    setEdges((current) =>
      current.map((edge) => {
        const targetStatus = subtaskStates[edge.target]?.status;
        return {
          ...edge,
          animated: targetStatus === "in_progress",
        };
      }),
    );
  }, [plan, setEdges, setNodes, subtaskStates]);

  if (!plan || plan.subtasks.length === 0) return null;

  return (
    <div className="space-y-3">
      {plan.planner_status === "direct" && (
        <div className="flex items-start gap-2 border border-blue-200 bg-blue-50 px-3 py-2 text-xs leading-5 text-blue-800 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
          <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <span>均衡模式判定该问题范围集中，直接使用单个研究任务。</span>
        </div>
      )}
      {plan.planner_status === "fallback" && (
        <div
          role="alert"
          className="flex items-start gap-2 border border-amber-300 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200"
        >
          <TriangleAlert
            className="mt-0.5 h-4 w-4 shrink-0"
            aria-hidden="true"
          />
          <span>
            Planner 规划失败，已降级为单任务
            {plan.planner_error ? `：${plan.planner_error}` : "。"}
          </span>
        </div>
      )}
      <div className="relative h-[460px] w-full overflow-hidden rounded-lg border border-border bg-surface">
        <button
          type="button"
          onClick={resetLayout}
          className="absolute right-3 top-3 z-10 inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-text-muted shadow-sm hover:bg-surface-alt hover:text-text"
          title="恢复自动布局"
          aria-label="恢复任务 DAG 自动布局"
        >
          <RotateCcw className="h-4 w-4" />
        </button>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onInit={(instance) => {
            flowRef.current = instance;
          }}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.25}
          maxZoom={1.5}
          nodesConnectable={false}
          deleteKeyCode={null}
        >
          <Background gap={24} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}
