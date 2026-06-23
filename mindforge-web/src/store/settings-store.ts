import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

export type LLMProvider = "openai" | "deepseek";

export interface SettingsState {
  llmProvider: LLMProvider;
  llmApiKey: string;
  retrievalTopK: number;
  rerankTopK: number;
  maxIterations: number;
  criticThreshold: number;
  loaded: boolean;

  setLLMProvider: (p: LLMProvider) => void;
  setLLMApiKey: (k: string) => void;
  setRetrievalTopK: (k: number) => void;
  setRerankTopK: (k: number) => void;
  setMaxIterations: (n: number) => void;
  setCriticThreshold: (n: number) => void;
  loadSettings: () => Promise<void>;
  saveSettings: () => Promise<boolean>;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      llmProvider: "deepseek",
      llmApiKey: "",
      retrievalTopK: 20,
      rerankTopK: 6,
      maxIterations: 8,
      criticThreshold: 7.0,
      loaded: false,

      setLLMProvider: (p) => set({ llmProvider: p }),
      setLLMApiKey: (k) => set({ llmApiKey: k }),
      setRetrievalTopK: (k) => set({ retrievalTopK: k }),
      setRerankTopK: (k) => set({ rerankTopK: k }),
      setMaxIterations: (n) => set({ maxIterations: n }),
      setCriticThreshold: (n) => set({ criticThreshold: n }),

      loadSettings: async () => {
        try {
          const res = await fetch(`${API_BASE}/settings`);
          if (res.ok) {
            const data = await res.json();
            set({
              llmProvider: data.llm_provider || "deepseek",
              llmApiKey: "", // never prefill key from backend (masked)
              loaded: true,
            });
          }
        } catch {
          // Offline — use localStorage values
          set({ loaded: true });
        }
      },

      saveSettings: async () => {
        const state = get();
        try {
          const res = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              llm_provider: state.llmProvider,
              deepseek_api_key: state.llmProvider === "deepseek" ? state.llmApiKey : "",
              openai_api_key: state.llmProvider === "openai" ? state.llmApiKey : "",
              embedding_provider: "openai",
            }),
          });
          return res.ok;
        } catch {
          return false;
        }
      },
    }),
    {
      name: "mindforge-settings",
      partialize: (state) => ({
        llmProvider: state.llmProvider,
        retrievalTopK: state.retrievalTopK,
        rerankTopK: state.rerankTopK,
        maxIterations: state.maxIterations,
        criticThreshold: state.criticThreshold,
        // llmApiKey is NOT persisted to localStorage (sent to backend)
      }),
    },
  ),
);
