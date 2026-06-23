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

  setTask: (task: string) => void;
  reset: () => void;
  handleEvent: (event: SSEEvent) => void;
  setStatus: (status: ResearchState["status"], error?: string) => void;
}

export const useResearchStore = create<ResearchState>((set) => ({
  status: "idle",
  error: null,
  task: "",
  plan: null,
  subtasks: {},
  synthesizing: false,
  criticScore: null,
  refineRound: 0,
  finalResult: null,

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
    }),

  setStatus: (status, error) => set({ status, error: error ?? null }),

  handleEvent: (event) => {
    switch (event.type) {
      case "plan_ready":
        set({
          plan: event.plan,
          subtasks: Object.fromEntries(
            event.plan.subtasks.map((s) => [s.task_id, s]),
          ),
        });
        break;

      case "subtask_start":
        set((s) => ({
          subtasks: {
            ...s.subtasks,
            [event.task_id]: {
              ...s.subtasks[event.task_id],
              status: "in_progress",
            },
          },
        }));
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
