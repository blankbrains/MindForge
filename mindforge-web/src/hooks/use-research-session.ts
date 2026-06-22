import { useCallback, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { createSSEConnection } from "@/lib/sse-parser";

const RESEARCH_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const store = useResearchStore();
  const addFromResearch = useHistoryStore((s) => s.addFromResearch);

  const startResearch = useCallback(
    (task: string) => {
      abortRef.current?.abort();
      store.reset();
      store.setTask(task);
      store.setStatus("streaming");

      // Timeout safety: abort after RESEARCH_TIMEOUT_MS
      const timeoutId = setTimeout(() => {
        store.setStatus("error", "研究超时（5 分钟），请尝试简化问题");
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
          store.setStatus("error", err.message);
        },
      );
    },
    [store, addFromResearch],
  );

  const cancelResearch = useCallback(() => {
    abortRef.current?.abort();
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
