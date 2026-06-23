import { useCallback, useEffect, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { createSSEConnection } from "@/lib/sse-parser";

const RESEARCH_TIMEOUT_MS = 15 * 60 * 1000; // 15 分钟，适配多轮精炼

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const researchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const store = useResearchStore();
  const addFromResearch = useHistoryStore((s) => s.addFromResearch);

  // 组件卸载时清理 SSE 连接和超时
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (researchTimeoutRef.current) {
        clearTimeout(researchTimeoutRef.current);
        researchTimeoutRef.current = null;
      }
    };
  }, []);

  const startResearch = useCallback(
    (task: string) => {
      if (researchTimeoutRef.current) { clearTimeout(researchTimeoutRef.current); researchTimeoutRef.current = null; }
      abortRef.current?.abort();

      // 使用 getState 确保拿到最新 setState action，避免闭包陈旧引用
      useResearchStore.setState({
        status: "streaming", error: null, plan: null, subtasks: {},
        synthesizing: false, criticScore: null, refineRound: 0,
      });
      useResearchStore.getState().setTask(task);

      const timeoutId = setTimeout(() => {
        useResearchStore.getState().setStatus(
          "error", "研究超时（15 分钟），请尝试简化问题"
        );
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
        () => {
          clearTimeout(timeoutId);
          // 仅当 done 事件已将 finalResult 写入后才置 completed，
          // 避免 [DONE] 标记先于 done 事件到达时出现"已完成但无报告"白屏
          if (useResearchStore.getState().finalResult) {
            useResearchStore.getState().setStatus("completed");
          }
        },
        (err) => {
          clearTimeout(timeoutId);
          const msg = err.message || "";
          if (err instanceof Error && "status" in err) {
            const status = (err as unknown as Record<string, unknown>).status as number;
            if (status === 401 || status === 403) {
              store.setStatus("error", "API Key 无效或已过期，请在设置中更新。");
              return;
            }
          }
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
    useResearchStore.getState().setStatus("idle");
  }, []);

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
