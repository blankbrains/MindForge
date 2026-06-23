import { useCallback, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { createSSEConnection } from "@/lib/sse-parser";

const RESEARCH_TIMEOUT_MS = 5 * 60 * 1000;

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const researchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const store = useResearchStore();
  const addFromResearch = useHistoryStore((s) => s.addFromResearch);

  const startResearch = useCallback(
    (task: string) => {
      if (researchTimeoutRef.current) { clearTimeout(researchTimeoutRef.current); researchTimeoutRef.current = null; }
      abortRef.current?.abort();

      // Only reset progress, keep task and finalResult visible
      useResearchStore.setState({
        status: "streaming", error: null, plan: null, subtasks: {},
        synthesizing: false, criticScore: null, refineRound: 0,
      });
      store.setTask(task);

      const timeoutId = setTimeout(() => {
        store.setStatus("error", "研究超时（5 分钟），请尝试简化问题或检查 API Key");
        abortRef.current?.abort();
      }, RESEARCH_TIMEOUT_MS);
      researchTimeoutRef.current = timeoutId;

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
        () => { clearTimeout(timeoutId); store.setStatus("completed"); },
        (err) => {
          clearTimeout(timeoutId);
          const msg = err.message || "";
          if (msg.includes("401") || msg.includes("403") || msg.includes("auth")) {
            store.setStatus("error", "API Key 无效或已过期，请在设置中更新。");
          } else if (msg.includes("timeout") || msg.includes("abort")) {
            store.setStatus("error", "研究超时，请尝试简化问题");
          } else {
            store.setStatus("error", `研究失败: ${msg}`);
          }
        },
      );
    },
    [store, addFromResearch],
  );

  const cancelResearch = useCallback(() => {
    abortRef.current?.abort();
    if (researchTimeoutRef.current) { clearTimeout(researchTimeoutRef.current); researchTimeoutRef.current = null; }
    store.setStatus("idle");
  }, [store]);

  return {
    ...store,
    startResearch,
    cancelResearch,
    isIdle: store.status === "idle",
    isStreaming: store.status === "streaming" || store.status === "connecting",
    isCompleted: store.status === "completed",
    isError: store.status === "error",
  };
}
