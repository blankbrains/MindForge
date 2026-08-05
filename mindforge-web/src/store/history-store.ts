import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";
import { normalizeCitationSources } from "@/lib/citations";
import type { CitationSource } from "@/types/research";

const LOCAL_HISTORY_ID_FLOOR = 1_000_000_000_000;
let historyLoadGeneration = 0;

function sortHistoryEntries(entries: HistoryEntry[]): HistoryEntry[] {
  return entries.sort((a, b) =>
    (b.created_at ?? "").localeCompare(a.created_at ?? ""),
  );
}

export interface HistoryEntry {
  id: number;
  task: string;
  report: string | null;
  quality_score: number | null;
  model_used: string | null;
  token_usage?: Record<string, unknown>;
  sources?: CitationSource[];
  trace_id?: string | null;
  conversation_id?: string | null;
  run_id?: string | null;
  created_at: string | null;
}

interface ResearchUsageSummary {
  tokenUsage?: Record<string, number>;
  costUsd?: number | null;
  costStatus?: string;
}

export interface HistoryState {
  entries: HistoryEntry[];
  localEntries: HistoryEntry[];
  loaded: boolean;
  loading: boolean;
  loadError: string | null;
  page: number;
  pageSize: number;
  total: number;
  serverTotal: number;

  addEntry: (entry: HistoryEntry) => void;
  addFromResearch: (
    task: string,
    report: string,
    quality?: number,
    model?: string,
    usage?: ResearchUsageSummary,
    sources?: CitationSource[],
    traceId?: string,
    conversationId?: string,
    runId?: string,
  ) => Promise<void>;
  loadHistory: (page?: number) => Promise<void>;
  loadEntry: (id: number) => Promise<void>;
  removeEntry: (id: number) => Promise<void>;
  clearAll: () => Promise<void>;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      entries: [],
      localEntries: [],
      loaded: false,
      loading: false,
      loadError: null,
      page: 1,
      pageSize: 20,
      total: 0,
      serverTotal: 0,

      addEntry: (entry) =>
        set((s) => {
          const localEntries =
            entry.id >= LOCAL_HISTORY_ID_FLOOR
              ? sortHistoryEntries([entry, ...s.localEntries]).slice(0, 100)
              : s.localEntries;
          const serverEntries =
            entry.id < LOCAL_HISTORY_ID_FLOOR
              ? [entry, ...s.entries.filter(
                  (current) => current.id < LOCAL_HISTORY_ID_FLOOR,
                )]
              : s.entries.filter(
                  (current) => current.id < LOCAL_HISTORY_ID_FLOOR,
                );
          return {
            localEntries,
            entries:
              s.page === 1
                ? sortHistoryEntries([
                    ...localEntries,
                    ...sortHistoryEntries(serverEntries).slice(0, s.pageSize),
                  ])
                : s.entries,
            total: s.total + 1,
            serverTotal:
              entry.id < LOCAL_HISTORY_ID_FLOOR
                ? s.serverTotal + 1
                : s.serverTotal,
          };
        }),

      addFromResearch: async (
        task,
        report,
        quality,
        model,
        usage,
        sources,
        traceId,
        conversationId,
        runId,
      ) => {
        const normalizedSources = normalizeCitationSources(sources);
        const tokenUsage = {
          ...(usage?.tokenUsage ?? {}),
          billing: {
            estimated_cost_usd: usage?.costUsd ?? null,
            status: usage?.costStatus ?? "usage_unavailable",
          },
        };
        let serverId: number | null = null;
        let acceptedByServer = false;
        try {
          const res = await fetch(`${API_BASE}/history`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              task,
              report,
              quality_score: quality ?? null,
              model_used: model ?? null,
              token_usage: tokenUsage,
              sources: normalizedSources,
              trace_id: traceId ?? null,
              conversation_id: conversationId ?? null,
              run_id: runId ?? null,
            }),
          });
          if (res.ok) {
            acceptedByServer = true;
            try {
              const data = await res.json() as { id?: number };
              serverId =
                typeof data.id === "number" ? data.id : null;
            } catch (error) {
              console.warn(
                "history-store: POST succeeded but response was invalid; reloading history",
                error,
              );
            }
          }
        } catch (e) {
          console.warn("history-store: POST failed, using local-only entry", e);
        }
        if (acceptedByServer && serverId === null) {
          await get().loadHistory();
          return;
        }
        const entry: HistoryEntry = {
          id: serverId ?? Math.floor(Date.now() * 1000 + Math.random() * 1000),
          task,
          report: report.slice(0, 3000),  // 列表预览用，完整报告由后端存储
          quality_score: quality ?? null,
          model_used: model ?? null,
          token_usage: tokenUsage,
          sources: normalizedSources,
          trace_id: traceId ?? null,
          conversation_id: conversationId ?? null,
          run_id: runId ?? null,
          created_at: new Date().toISOString(),
        };
        get().addEntry(entry);
      },

      loadHistory: async (requestedPage = 1) => {
        const page = Math.max(1, Math.floor(requestedPage));
        const pageSize = get().pageSize;
        const generation = historyLoadGeneration + 1;
        historyLoadGeneration = generation;
        set({ loading: true, loadError: null });
        try {
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
          const serverEntries = data.entries ?? [];
          const serverTotal = data.total ?? serverEntries.length;
          const lastPage = Math.max(1, Math.ceil(serverTotal / pageSize));
          if (page > lastPage) {
            if (historyLoadGeneration !== generation) return;
            await get().loadHistory(lastPage);
            return;
          }
          if (historyLoadGeneration !== generation) return;

          const localEntries =
            page === 1
              ? get().localEntries
              : [];
          const merged = sortHistoryEntries([
            ...localEntries,
            ...serverEntries,
          ]);
          set({
            entries: merged,
            page,
            total: serverTotal + get().localEntries.length,
            serverTotal,
            loaded: true,
            loading: false,
            loadError: null,
          });
        } catch (error) {
          console.warn("history-store: history load failed", error);
          if (historyLoadGeneration !== generation) return;
          set({
            loaded: true,
            loading: false,
            loadError:
              error instanceof Error
                ? error.message
                : "历史记录加载失败。",
          });
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
          await get().loadHistory(get().page);
          return;
        }
        set((s) => ({
          entries: s.entries.filter((e) => e.id !== id),
          localEntries: s.localEntries.filter((e) => e.id !== id),
          total: Math.max(0, s.total - 1),
          serverTotal:
            id < LOCAL_HISTORY_ID_FLOOR
              ? Math.max(0, s.serverTotal - 1)
              : s.serverTotal,
        }));
      },

      clearAll: async () => {
        const res = await fetch(
          `${API_BASE}/history`,
          { method: "DELETE" },
        );
        if (!res.ok) {
          throw new Error(
            `History clear failed with ${res.status}`,
          );
        }
        set({
          entries: [],
          localEntries: [],
          page: 1,
          total: 0,
          serverTotal: 0,
          loadError: null,
        });
      },
    }),
    {
      name: "mindforge-history",
      partialize: (state) => ({
        localEntries: state.localEntries,
      }),
    },
  ),
);
