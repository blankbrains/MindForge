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

interface ResearchState {
  status: "idle" | "connecting" | "streaming" | "completed" | "error";
  error: string | null;
  task: string;
  plan: ResearchPlan | null;
  subtasks: SubTaskState;
  synthesizing: boolean;
  criticScore: CriticScore | null;
  refineRound: number;
  finalResult: AgentResult | null;
  streamingAnswer: string;

  setTask: (task: string) => void;
  reset: () => void;
  handleEvent: (event: SSEEvent) => void;
  setStatus: (status: ResearchState["status"], error?: string) => void;
}

export const useResearchStore = create<ResearchState>((set, get) => ({
  status: "idle",
  error: null,
  task: "",
  plan: null,
  subtasks: {},
  synthesizing: false,
  criticScore: null,
  refineRound: 0,
  finalResult: null,
  streamingAnswer: "",

  setTask: (task) => set({ task }),

  reset: () =>
    set({
      status: "idle",
      error: null,
      task: "",
      plan: null,
      subtasks: {},
      synthesizing: false,
      criticScore: null,
      refineRound: 0,
      finalResult: null,
      streamingAnswer: "",
    }),

  setStatus: (status, error) => set({ status, error: error ?? null }),

  handleEvent: (event) => {
    switch (event.type) {
      case "plan_ready":
        // 幂等防护：重复 plan_ready 不覆盖已有进度
        if (get().plan) break;
        set({
          plan: event.plan,
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
        set((s) => ({
          subtasks: {
            ...s.subtasks,
            [event.task_id]: {
              ...s.subtasks[event.task_id],
              status: event.result.success ? "completed" : "failed",
              result: event.result,
            },
          },
        }));
        break;

      case "synthesizing":
        set({ synthesizing: event.status === "start" });
        break;

      case "critic_feedback":
        set({ criticScore: event.score });
        break;

      case "refining":
        set({ refineRound: event.round });
        break;

      case "answer_chunk":
        set((s) => ({ streamingAnswer: s.streamingAnswer + event.content }));
        break;

      case "done":
        set({
          finalResult: event.result,
          status: "completed",
          synthesizing: false,
          refineRound: 0,
        });
        break;

      default:
        break;
    }
  },
}));
