import { create } from "zustand";
import type { AgentResult } from "@/types/research";

export interface HistoryEntry {
  id: string;
  task: string;
  result: AgentResult;
  createdAt: number; // Date.now() timestamp
}

interface HistoryState {
  entries: HistoryEntry[];

  addEntry: (task: string, result: AgentResult) => void;
  removeEntry: (id: string) => void;
  clearAll: () => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  entries: [],

  addEntry: (task, result) =>
    set((s) => ({
      entries: [
        {
          id: crypto.randomUUID(),
          task,
          result,
          createdAt: Date.now(),
        },
        ...s.entries,
      ].slice(0, 100), // cap at 100 entries
    })),

  removeEntry: (id) =>
    set((s) => ({
      entries: s.entries.filter((e) => e.id !== id),
    })),

  clearAll: () => set({ entries: [] }),
}));
