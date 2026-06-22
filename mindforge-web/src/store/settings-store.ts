import { create } from "zustand";

export type LLMProvider = "openai" | "deepseek";

export interface SettingsState {
  // LLM
  llmProvider: LLMProvider;
  llmApiKey: string;

  // Retrieval
  retrievalTopK: number;
  rerankTopK: number;

  // Agent
  maxIterations: number;
  criticThreshold: number;

  // Actions
  setLLMProvider: (p: LLMProvider) => void;
  setLLMApiKey: (k: string) => void;
  setRetrievalTopK: (k: number) => void;
  setRerankTopK: (k: number) => void;
  setMaxIterations: (n: number) => void;
  setCriticThreshold: (n: number) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  llmProvider: "deepseek",
  llmApiKey: "",
  retrievalTopK: 20,
  rerankTopK: 6,
  maxIterations: 8,
  criticThreshold: 7.0,

  setLLMProvider: (p) => set({ llmProvider: p }),
  setLLMApiKey: (k) => set({ llmApiKey: k }),
  setRetrievalTopK: (k) => set({ retrievalTopK: k }),
  setRerankTopK: (k) => set({ rerankTopK: k }),
  setMaxIterations: (n) => set({ maxIterations: n }),
  setCriticThreshold: (n) => set({ criticThreshold: n }),
}));
