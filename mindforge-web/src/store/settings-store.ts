import { create } from "zustand";
import { persist } from "zustand/middleware";
import { API_BASE } from "@/lib/constants";

export type LLMProvider = "openai" | "deepseek";

export interface SettingsState {
  llmProvider: LLMProvider;
  llmApiKey: string;
  hasLLMKey: boolean;  // 后端是否已保存 key（从 masked 值判断）
  retrievalTopK: number;
  rerankTopK: number;
  maxIterations: number;
  criticThreshold: number;
  loaded: boolean;

  setLLMProvider: (p: LLMProvider) => void;
  setLLMApiKey: (k: string) => void;
  clearLLMApiKey: () => void;
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
      hasLLMKey: false,
      retrievalTopK: 20,
      rerankTopK: 6,
      maxIterations: 8,
      criticThreshold: 7.0,
      loaded: false,

      setLLMProvider: (p) => set({ llmProvider: p }),
      setLLMApiKey: (k) => set({ llmApiKey: k, hasLLMKey: k.length > 0 }),
      clearLLMApiKey: () => set({ llmApiKey: "", hasLLMKey: false }),
      setRetrievalTopK: (k) => set({ retrievalTopK: k }),
      setRerankTopK: (k) => set({ rerankTopK: k }),
      setMaxIterations: (n) => set({ maxIterations: n }),
      setCriticThreshold: (n) => set({ criticThreshold: n }),

      loadSettings: async () => {
        try {
          const res = await fetch(`${API_BASE}/settings`);
          if (res.ok) {
            const data = await res.json();
            // 后端返回的 masked key：non-empty 表示已配置，empty 表示未配置
            const maskedKey =
              get().llmProvider === "deepseek"
                ? (data.deepseek_api_key || "")
                : (data.openai_api_key || "");
            const hasKey = maskedKey.length > 0;
            set({
              llmProvider: data.llm_provider || "deepseek",
              // 如果后端已有 key，显示脱敏值；否则保留 localStorage 值
              llmApiKey: hasKey ? maskedKey : get().llmApiKey,
              hasLLMKey: hasKey,
              loaded: true,
            });
          }
        } catch {
          set({ loaded: true });
        }
      },

      saveSettings: async () => {
        const state = get();
        try {
          // 脱敏 key 不发送（保护已配置的 key）；用户输入新 key 时正常发送；空字符串 = 删除
          const isMasked = state.llmApiKey.startsWith("***");
          const deepseekKey = isMasked ? undefined : (state.llmProvider === "deepseek" ? state.llmApiKey : "");
          const openaiKey = isMasked ? undefined : (state.llmProvider === "openai" ? state.llmApiKey : "");
          const res = await fetch(`${API_BASE}/settings`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              llm_provider: state.llmProvider,
              deepseek_api_key: deepseekKey,
              openai_api_key: openaiKey,
              embedding_provider: state.llmProvider === "openai" ? "openai" : "bge",
              retrieval_top_k: state.retrievalTopK,
              rerank_top_k: state.rerankTopK,
              max_iterations: state.maxIterations,
              critic_threshold: state.criticThreshold,
            }),
          });
          if (res.ok) {
            // 保存成功后刷新，确保 hasLLMKey 与后端一致
            await get().loadSettings();
            return true;
          }
          return false;
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
        retrievalTopK: state.retrievalTopK,
        rerankTopK: state.rerankTopK,
        maxIterations: state.maxIterations,
        criticThreshold: state.criticThreshold,
        // llmApiKey is NOT persisted (security); hasLLMKey persists for UI gating
      }),
    },
  ),
);
