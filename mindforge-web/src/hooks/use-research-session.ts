import { useCallback, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { createSSEConnection } from "@/lib/sse-parser";

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const store = useResearchStore();

  const startResearch = useCallback(
    (task: string) => {
      store.setTask(task);
      store.setStatus("connecting");
      abortRef.current?.abort();

      store.reset();
      store.setTask(task);
      store.setStatus("streaming");

      abortRef.current = createSSEConnection<SSEEvent>(
        `${API_BASE}/query`,
        { task, stream: true },
        (event) => store.handleEvent(event),
        () => store.setStatus("completed"),
        (err) => store.setStatus("error", err.message),
      );
    },
    [store],
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
