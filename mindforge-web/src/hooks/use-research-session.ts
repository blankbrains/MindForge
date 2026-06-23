import { useCallback, useRef, useState } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { useSettingsStore } from "@/store/settings-store";
import { createSSEConnection } from "@/lib/sse-parser";

const RESEARCH_TIMEOUT_MS = 5 * 60 * 1000;

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const store = useResearchStore();
  const addFromResearch = useHistoryStore((s) => s.addFromResearch);
  const hasApiKey = useSettingsStore((s) => !!s.llmApiKey);
  const [docOnlyResult, setDocOnlyResult] = useState<{
    output: string; quality?: number;
  } | null>(null);

  /** Quick doc-search without LLM (no API key needed). */
  const startDocSearch = useCallback(
    async (task: string) => {
      store.reset();
      store.setTask(task);
      store.setStatus("streaming");
      const timeoutId = setTimeout(() => {
        store.setStatus("error", "检索超时（30 秒）");
      }, 30_000);

      try {
        const res = await fetch(`${API_BASE}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task, stream: false }),
        });
        clearTimeout(timeoutId);
        if (res.ok) {
          const data = await res.json();
          const report = data.report || "";
          setDocOnlyResult({ output: report, quality: data.quality_score });
          store.setStatus("completed");
          addFromResearch(task, report, data.quality_score);
        } else {
          const text = await res.text();
          store.setStatus("error", text || "检索失败");
        }
      } catch (err: unknown) {
        clearTimeout(timeoutId);
        store.setStatus("error", err instanceof Error ? err.message : "网络错误");
      }
    },
    [store, addFromResearch],
  );

  /** Full Agent research — requires LLM API key. */
  const startResearch = useCallback(
    (task: string) => {
      if (!hasApiKey) {
        // No API key — fall back to doc-only search
        startDocSearch(task);
        return;
      }

      abortRef.current?.abort();
      store.reset();
      store.setTask(task);
      store.setStatus("streaming");

      const timeoutId = setTimeout(() => {
        store.setStatus("error", "研究超时（5 分钟），请尝试简化问题或检查 API Key");
        abortRef.current?.abort();
      }, RESEARCH_TIMEOUT_MS);

      abortRef.current = createSSEConnection<SSEEvent>(
        `${API_BASE}/query`,
        { task, stream: true },
        (event) => {
          store.handleEvent(event);
          if (event.type === "done") {
            clearTimeout(timeoutId);
            const result = event.result as unknown as Record<string, unknown> | undefined;
            const report = (result?.output as string) || "";
            const quality = (result?.metadata as Record<string, unknown> | undefined)?.quality as number | undefined;
            addFromResearch(task, report, quality);
          }
        },
        () => {
          clearTimeout(timeoutId);
          store.setStatus("completed");
        },
        (err) => {
          clearTimeout(timeoutId);
          const msg = err.message || "";
          if (msg.includes("401") || msg.includes("403") || msg.includes("auth")) {
            store.setStatus("error", "API Key 无效或已过期，请在设置中更新。没有 API Key 也可以使用文档检索。");
          } else if (msg.includes("timeout") || msg.includes("abort")) {
            store.setStatus("error", "研究超时，请尝试简化问题");
          } else {
            store.setStatus("error", `研究失败: ${msg}`);
          }
        },
      );
    },
    [store, addFromResearch, hasApiKey, startDocSearch],
  );

  const cancelResearch = useCallback(() => {
    abortRef.current?.abort();
    store.setStatus("idle");
  }, [store]);

  return {
    ...store,
    startResearch,
    startDocSearch,
    cancelResearch,
    docOnlyResult,
    hasApiKey,
    isIdle: store.status === "idle",
    isStreaming: store.status === "streaming" || store.status === "connecting",
    isCompleted: store.status === "completed",
    isError: store.status === "error",
  };
}
