import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

const LOCAL_HISTORY_ID_FLOOR = 1_000_000_000_000;

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
  loadEntry: (id: number) => Promise<void>;
  removeEntry: (id: number) => Promise<void>;
  clearAll: () => Promise<void>;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      entries: [],
      loaded: false,

      addEntry: (entry) =>
        set((s) => ({
          entries: [entry, ...s.entries]
            .sort((a, b) => ((b.created_at ?? 0) > (a.created_at ?? 0) ? 1 : -1))
            .slice(0, 100),
        })),

      addFromResearch: async (task, report, quality, model) => {
        let serverId: number | null = null;
        try {
          const res = await fetch(`${API_BASE}/history`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              task,
              report,
              quality_score: quality ?? null,
              model_used: model ?? null,
            }),
          });
          if (res.ok) {
            const data = await res.json() as { id?: number };
            serverId = data.id ?? null;
          }
        } catch (e) {
          console.warn("history-store: POST failed, using local-only entry", e);
        }
        const entry: HistoryEntry = {
          id: serverId ?? Math.floor(Date.now() * 1000 + Math.random() * 1000),
          task,
          report: report.slice(0, 3000),  // 列表预览用，完整报告由后端存储
          quality_score: quality ?? null,
          model_used: model ?? null,
          created_at: new Date().toISOString(),
        };
        get().addEntry(entry);
      },

      loadHistory: async () => {
        try {
          const pageSize = 100;
          let page = 1;
          let total = 0;
          const serverEntries: HistoryEntry[] = [];

          do {
            const res = await fetch(
              `${API_BASE}/history?page=${page}&page_size=${pageSize}`,
            );
            if (!res.ok) {
              throw new Error(
                `History request failed with ${res.status}`,
              );
            }
            const data = await res.json() as {
              entries?: HistoryEntry[];
              total?: number;
            };
            serverEntries.push(...(data.entries ?? []));
            total = data.total ?? serverEntries.length;
            page += 1;
          } while (
            serverEntries.length < total
            && page <= Math.ceil(total / pageSize) + 1
          );

          const localEntries = get().entries.filter(
            (entry) => entry.id >= LOCAL_HISTORY_ID_FLOOR,
          );
          const merged = [...localEntries, ...serverEntries]
            .sort((a, b) =>
              (b.created_at ?? "").localeCompare(a.created_at ?? ""),
            );
          set({ entries: merged });
        } catch (error) {
          console.warn("history-store: history load failed", error);
        } finally {
          set({ loaded: true });
        }
      },

      loadEntry: async (id: number) => {
        if (id >= LOCAL_HISTORY_ID_FLOOR) return;
        const res = await fetch(`${API_BASE}/history/${id}`);
        if (!res.ok) {
          throw new Error(
            `History detail request failed with ${res.status}`,
          );
        }
        const entry = await res.json() as HistoryEntry;
        set((state) => ({
          entries: state.entries.map((current) =>
            current.id === id ? entry : current,
          ),
        }));
      },

      removeEntry: async (id: number) => {
        if (id < LOCAL_HISTORY_ID_FLOOR) {
          const res = await fetch(
            `${API_BASE}/history/${id}`,
            { method: "DELETE" },
          );
          if (!res.ok) {
            throw new Error(
              `History deletion failed with ${res.status}`,
            );
          }
        }
        set((s) => ({ entries: s.entries.filter((e) => e.id !== id) }));
      },

      clearAll: async () => {
        const hasServerEntries = get().entries.some(
          (entry) => entry.id < LOCAL_HISTORY_ID_FLOOR,
        );
        if (hasServerEntries) {
          const res = await fetch(
            `${API_BASE}/history`,
            { method: "DELETE" },
          );
          if (!res.ok) {
            throw new Error(
              `History clear failed with ${res.status}`,
            );
          }
        }
        set({ entries: [] });
      },
    }),
    {
      name: "mindforge-history",
    },
  ),
);
