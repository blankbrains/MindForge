import { useCallback, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { createSSEConnection } from "@/lib/sse-parser";

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const store = useResearchStore();
  const addHistory = useHistoryStore((s) => s.addEntry);

  const startResearch = useCallback(
    (task: string) => {
      abortRef.current?.abort();
      store.reset();
      store.setTask(task);
      store.setStatus("streaming");

      const taskSnapshot = task;

      abortRef.current = createSSEConnection<SSEEvent>(
        `${API_BASE}/query`,
        { task, stream: true },
        (event) => {
          store.handleEvent(event);
          if (event.type === "done") {
            addHistory(taskSnapshot, event.result);
          }
        },
        () => store.setStatus("completed"),
        (err) => store.setStatus("error", err.message),
      );
    },
    [store, addHistory],
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
