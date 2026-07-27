import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

export type LLMProvider = "openai" | "deepseek";

type ProviderValues = Record<LLMProvider, string>;
type ProviderFlags = Record<LLMProvider, boolean>;

interface SettingsPayload {
  llm_provider?: LLMProvider;
  deepseek_api_key?: string;
  openai_api_key?: string;
  embedding_provider?: "openai" | "bge";
  retrieval_top_k?: number;
  rerank_top_k?: number;
  max_iterations?: number;
  max_refine_rounds?: number;
  critic_threshold?: number;
  subtask_timeout?: number;
  research_timeout?: number;
}

export interface SettingsState {
  llmProvider: LLMProvider;
  llmApiKey: string;
  hasLLMKey: boolean;
  maskedKeys: ProviderValues;
  apiKeyDrafts: Partial<ProviderValues>;
  hasLLMKeys: ProviderFlags;
  retrievalTopK: number;
  rerankTopK: number;
  maxIterations: number;
  maxRefineRounds: number;
  criticThreshold: number;
  subtaskTimeout: number;
  researchTimeout: number;
  loaded: boolean;

  setLLMProvider: (provider: LLMProvider) => void;
  setLLMApiKey: (key: string) => void;
  clearLLMApiKey: () => void;
  restoreLLMApiKey: () => void;
  setRetrievalTopK: (value: number) => void;
  setRerankTopK: (value: number) => void;
  setMaxIterations: (value: number) => void;
  setMaxRefineRounds: (value: number) => void;
  setCriticThreshold: (value: number) => void;
  setSubtaskTimeout: (value: number) => void;
  setResearchTimeout: (value: number) => void;
  loadSettings: () => Promise<void>;
  saveSettings: () => Promise<boolean>;
  deleteLLMApiKey: () => Promise<boolean>;
}

const EMPTY_KEYS: ProviderValues = { openai: "", deepseek: "" };
const EMPTY_FLAGS: ProviderFlags = { openai: false, deepseek: false };

function providerKeyName(provider: LLMProvider) {
  return provider === "deepseek" ? "deepseek_api_key" : "openai_api_key";
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      llmProvider: "deepseek",
      llmApiKey: "",
      hasLLMKey: false,
      maskedKeys: { ...EMPTY_KEYS },
      apiKeyDrafts: {},
      hasLLMKeys: { ...EMPTY_FLAGS },
      retrievalTopK: 20,
      rerankTopK: 6,
      maxIterations: 3,
      maxRefineRounds: 1,
      criticThreshold: 7,
      subtaskTimeout: 30,
      researchTimeout: 180,
      loaded: false,

      setLLMProvider: (provider) => {
        const state = get();
        const displayedKey =
          state.apiKeyDrafts[provider] ?? state.maskedKeys[provider];
        set({
          llmProvider: provider,
          llmApiKey: displayedKey,
          hasLLMKey: state.hasLLMKeys[provider],
        });
      },

      setLLMApiKey: (key) =>
        set((state) => ({
          llmApiKey: key,
          apiKeyDrafts: {
            ...state.apiKeyDrafts,
            [state.llmProvider]: key,
          },
        })),

      clearLLMApiKey: () =>
        set((state) => ({
          llmApiKey: "",
          maskedKeys: {
            ...state.maskedKeys,
            [state.llmProvider]: "",
          },
          apiKeyDrafts: {
            ...state.apiKeyDrafts,
            [state.llmProvider]: "",
          },
          hasLLMKey: false,
          hasLLMKeys: {
            ...state.hasLLMKeys,
            [state.llmProvider]: false,
          },
        })),

      restoreLLMApiKey: () =>
        set((state) => {
          const drafts = { ...state.apiKeyDrafts };
          delete drafts[state.llmProvider];
          return {
            apiKeyDrafts: drafts,
            llmApiKey: state.maskedKeys[state.llmProvider],
            hasLLMKey: state.hasLLMKeys[state.llmProvider],
          };
        }),

      setRetrievalTopK: (value) => set({ retrievalTopK: value }),
      setRerankTopK: (value) => set({ rerankTopK: value }),
      setMaxIterations: (value) => set({ maxIterations: value }),
      setMaxRefineRounds: (value) => set({ maxRefineRounds: value }),
      setCriticThreshold: (value) => set({ criticThreshold: value }),
      setSubtaskTimeout: (value) => set({ subtaskTimeout: value }),
      setResearchTimeout: (value) => set({ researchTimeout: value }),

      loadSettings: async () => {
        try {
          const response = await fetch(`${API_BASE}/settings`);
          if (!response.ok) {
            throw new Error(`Settings request failed with ${response.status}`);
          }

          const data = (await response.json()) as SettingsPayload;
          const provider = data.llm_provider ?? "deepseek";
          const maskedKeys: ProviderValues = {
            deepseek: data.deepseek_api_key ?? "",
            openai: data.openai_api_key ?? "",
          };
          const hasLLMKeys: ProviderFlags = {
            deepseek: maskedKeys.deepseek.length > 0,
            openai: maskedKeys.openai.length > 0,
          };

          set({
            llmProvider: provider,
            llmApiKey: maskedKeys[provider],
            hasLLMKey: hasLLMKeys[provider],
            maskedKeys,
            apiKeyDrafts: {},
            hasLLMKeys,
            retrievalTopK: data.retrieval_top_k ?? 20,
            rerankTopK: data.rerank_top_k ?? 6,
            maxIterations: data.max_iterations ?? 3,
            maxRefineRounds: data.max_refine_rounds ?? 1,
            criticThreshold: data.critic_threshold ?? 7,
            subtaskTimeout: data.subtask_timeout ?? 30,
            researchTimeout: data.research_timeout ?? 180,
            loaded: true,
          });
        } catch {
          set({ loaded: true });
        }
      },

      saveSettings: async () => {
        const state = get();
        const payload: SettingsPayload = {
          llm_provider: state.llmProvider,
          embedding_provider:
            state.llmProvider === "openai" ? "openai" : "bge",
          retrieval_top_k: state.retrievalTopK,
          rerank_top_k: state.rerankTopK,
          max_iterations: state.maxIterations,
          max_refine_rounds: state.maxRefineRounds,
          critic_threshold: state.criticThreshold,
          subtask_timeout: state.subtaskTimeout,
          research_timeout: state.researchTimeout,
        };

        if (!state.llmApiKey.startsWith("***")) {
          payload[providerKeyName(state.llmProvider)] = state.llmApiKey;
        }

        try {
          const response = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) return false;
          await get().loadSettings();
          return true;
        } catch {
          return false;
        }
      },

      deleteLLMApiKey: async () => {
        const state = get();
        const payload: SettingsPayload = {
          llm_provider: state.llmProvider,
          embedding_provider:
            state.llmProvider === "openai" ? "openai" : "bge",
          retrieval_top_k: state.retrievalTopK,
          rerank_top_k: state.rerankTopK,
          max_iterations: state.maxIterations,
          max_refine_rounds: state.maxRefineRounds,
          critic_threshold: state.criticThreshold,
          subtask_timeout: state.subtaskTimeout,
          research_timeout: state.researchTimeout,
          [providerKeyName(state.llmProvider)]: "",
        };
        try {
          const response = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) return false;
          await get().loadSettings();
          return true;
        } catch {
          return false;
        }
      },
    }),
    {
      name: "mindforge-settings",
      partialize: (state) => ({
        llmProvider: state.llmProvider,
        hasLLMKey: state.hasLLMKey,
        hasLLMKeys: state.hasLLMKeys,
        retrievalTopK: state.retrievalTopK,
        rerankTopK: state.rerankTopK,
        maxIterations: state.maxIterations,
        maxRefineRounds: state.maxRefineRounds,
        criticThreshold: state.criticThreshold,
        subtaskTimeout: state.subtaskTimeout,
        researchTimeout: state.researchTimeout,
      }),
    },
  ),
);
