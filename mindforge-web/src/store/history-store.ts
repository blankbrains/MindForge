import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

export interface HistoryEntry {
  id: number;
  task: string;
  report: string | null;
  quality_score: number | null;
  model_used: string | null;
  created_at: string | null;
}

export interface HistoryState {
  entries: HistoryEntry[];
  loaded: boolean;

  addEntry: (entry: HistoryEntry) => void;
  addFromResearch: (task: string, report: string, quality?: number, model?: string) => Promise<void>;
  loadHistory: () => Promise<void>;
  clearAll: () => void;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      entries: [],
      loaded: false,

      addEntry: (entry) =>
        set((s) => ({ entries: [entry, ...s.entries].slice(0, 100) })),

      addFromResearch: async (task, report, quality, model) => {
        try {
          await fetch(`${API_BASE}/history`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              task,
              report,
              quality_score: quality ?? null,
              model_used: model ?? null,
            }),
          });
        } catch {
          // Best-effort: network may be down
        }
        const entry: HistoryEntry = {
          id: Date.now(),
          task,
          report: report.slice(0, 500),
          quality_score: quality ?? null,
          model_used: model ?? null,
          created_at: new Date().toISOString(),
        };
        get().addEntry(entry);
      },

      loadHistory: async () => {
        try {
          const res = await fetch(`${API_BASE}/history`);
          if (res.ok) {
            const data = await res.json();
            set({ entries: data.entries || [], loaded: true });
          }
        } catch {
          set({ loaded: true });
        }
      },

      clearAll: () => set({ entries: [] }),
    }),
    {
      name: "mindforge-history",
    },
  ),
);
