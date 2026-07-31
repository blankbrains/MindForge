import { create } from "zustand";
import type {
  ResearchPlan,
  SubTask,
  AgentResult,
  CriticScore,
  SSEEvent,
} from "@/types/research";

interface SubTaskState {
  [taskId: string]: SubTask & { result?: AgentResult };
}

function boundedScore(value: unknown): number {
  const score = typeof value === "number" ? value : Number(value);
  return Number.isFinite(score) ? Math.min(10, Math.max(0, score)) : 0;
}

function boundedTextList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.slice(0, 2_000))
    .filter((item) => item.trim().length > 0)
    .slice(0, 20);
}

function normalizeCriticScore(value: unknown): CriticScore | null {
  if (!value || typeof value !== "object") return null;
  const score = value as Record<string, unknown>;
  return {
    overall: boundedScore(score.overall),
    completeness: boundedScore(score.completeness),
    accuracy: boundedScore(score.accuracy),
    depth: boundedScore(score.depth),
    clarity: boundedScore(score.clarity),
    citations: boundedScore(score.citations),
    issues: boundedTextList(score.issues),
    suggestions: boundedTextList(score.suggestions),
    should_refine: score.should_refine === true,
    evaluation_status:
      score.evaluation_status === "failed" ? "failed" : "evaluated",
    evaluation_error:
      typeof score.evaluation_error === "string"
        ? score.evaluation_error.slice(0, 1_000)
        : null,
  };
}

interface ResearchState {
  status:
    | "idle"
    | "connecting"
    | "streaming"
    | "completed"
    | "cancelled"
    | "error";
  error: string | null;
  task: string;
  activeTask: string;
  plan: ResearchPlan | null;
  subtasks: SubTaskState;
  planning: boolean;
  synthesizing: boolean;
  criticScore: CriticScore | null;
  refineRound: number;
  finalResult: AgentResult | null;
  streamingAnswer: string;
  traceId: string | null;
  phase: string;
  startedAt: number | null;
  lastHeartbeatAt: number | null;

  setTask: (task: string) => void;
  reset: () => void;
  interrupt: (status?: "idle" | "cancelled") => void;
  fail: (error: string) => void;
  handleEvent: (event: SSEEvent) => void;
}

export const useResearchStore = create<ResearchState>((set, get) => ({
  status: "idle",
  error: null,
  task: "",
  activeTask: "",
  plan: null,
  subtasks: {},
  planning: false,
  synthesizing: false,
  criticScore: null,
  refineRound: 0,
  finalResult: null,
  streamingAnswer: "",
  traceId: null,
  phase: "idle",
  startedAt: null,
  lastHeartbeatAt: null,

  setTask: (task) => set({ task }),

  reset: () =>
    set({
      status: "idle",
      error: null,
      task: "",
      activeTask: "",
      plan: null,
      subtasks: {},
      planning: false,
      synthesizing: false,
      criticScore: null,
      refineRound: 0,
      finalResult: null,
      streamingAnswer: "",
      traceId: null,
      phase: "idle",
      startedAt: null,
      lastHeartbeatAt: null,
    }),

  interrupt: (status = "idle") =>
    set({
      status,
      error: null,
      activeTask: "",
      plan: null,
      subtasks: {},
      planning: false,
      synthesizing: false,
      criticScore: null,
      refineRound: 0,
      finalResult: null,
      streamingAnswer: "",
      traceId: null,
      phase: status,
      startedAt: null,
      lastHeartbeatAt: null,
    }),

  fail: (error) =>
    set({
      status: "error",
      error,
      activeTask: "",
      planning: false,
      synthesizing: false,
      phase: "failed",
      startedAt: null,
      lastHeartbeatAt: null,
    }),

  handleEvent: (event) => {
    if (event.trace_id) {
      set({ traceId: event.trace_id });
    }
    switch (event.type) {
      case "trace_started":
        set({
          phase: "starting",
          startedAt: get().startedAt ?? Date.now(),
        });
        break;

      case "planning":
        set({
          planning: event.status === "start",
          phase: event.status === "start" ? "planning" : "researching",
        });
        break;

      case "heartbeat":
        set({ lastHeartbeatAt: event.timestamp * 1000 });
        break;

      case "plan_ready":
        // 幂等防护：重复 plan_ready 不覆盖已有进度
        if (get().plan) break;
        set({
          plan: event.plan,
          planning: false,
          phase: "researching",
          subtasks: Object.fromEntries(
            event.plan.subtasks.map((s) => [s.task_id, s]),
          ),
        });
        break;

      case "subtask_start":
        set((s) => {
          // 防护：未知 task_id 忽略，避免创建残缺条目
          if (!s.subtasks[event.task_id]) return s;
          return {
            subtasks: {
              ...s.subtasks,
              [event.task_id]: {
                ...s.subtasks[event.task_id],
                status: "in_progress",
              },
            },
          };
        });
        break;

      case "subtask_result":
        set((s) => {
          if (!s.subtasks[event.task_id]) return s;
          return {
            phase: "researching",
            subtasks: {
              ...s.subtasks,
              [event.task_id]: {
                ...s.subtasks[event.task_id],
                status: event.result.success ? "completed" : "failed",
                result: event.result,
              },
            },
          };
        });
        break;

      case "synthesizing":
        set({
          synthesizing: event.status === "start",
          phase: event.status === "start" ? "synthesizing" : "reviewing",
        });
        break;

      case "critic_feedback":
        set({
          criticScore: normalizeCriticScore(event.score),
          phase: "reviewing",
        });
        break;

      case "refining":
        set({ refineRound: event.round, phase: "refining" });
        break;

      case "answer_chunk":
        set((s) => ({ streamingAnswer: s.streamingAnswer + event.content }));
        break;

      case "done":
        set({
          task:
            event.result.success && get().task === get().activeTask
              ? ""
              : get().task,
          finalResult: event.result,
          status: event.result.success ? "completed" : "error",
          error: event.result.success
            ? null
            : event.result.output || "研究任务执行失败",
          synthesizing: false,
          planning: false,
          refineRound: 0,
          activeTask: "",
          traceId: event.result.trace_id ?? event.trace_id ?? get().traceId,
          phase: event.result.success ? "completed" : "failed",
          startedAt: null,
          lastHeartbeatAt: null,
        });
        break;

      case "error":
        set({
          status: "error",
          error: event.content || "研究任务执行失败",
          activeTask: "",
          planning: false,
          synthesizing: false,
          phase: "failed",
          startedAt: null,
          lastHeartbeatAt: null,
        });
        break;

      default:
        break;
    }
  },
}));
