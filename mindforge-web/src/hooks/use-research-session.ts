import { useCallback, useEffect, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { createSSEConnection } from "@/lib/sse-parser";

const configuredResearchTimeout = Number.parseInt(
  import.meta.env.VITE_RESEARCH_TIMEOUT_MS || "",
  10,
);
const RESEARCH_TIMEOUT_MS =
  Number.isFinite(configuredResearchTimeout) && configuredResearchTimeout > 0
    ? configuredResearchTimeout
    : 15 * 60 * 1000;
const configuredStreamFlushMs = Number.parseInt(
  import.meta.env.VITE_STREAM_FLUSH_MS || "",
  10,
);
const STREAM_FLUSH_MS =
  Number.isFinite(configuredStreamFlushMs) && configuredStreamFlushMs > 0
    ? configuredStreamFlushMs
    : 50;
const configuredMaxStreamedAnswerChars = Number.parseInt(
  import.meta.env.VITE_MAX_STREAMED_ANSWER_CHARS || "",
  10,
);
const MAX_STREAMED_ANSWER_CHARS =
  Number.isFinite(configuredMaxStreamedAnswerChars) &&
  configuredMaxStreamedAnswerChars > 0
    ? configuredMaxStreamedAnswerChars
    : 1_000_000;

export function useResearchSession() {
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const researchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamFlushRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingAnswerRef = useRef("");
  const store = useResearchStore();
  const addFromResearch = useHistoryStore((s) => s.addFromResearch);

  const flushPendingAnswer = useCallback(() => {
    if (streamFlushRef.current) {
      clearTimeout(streamFlushRef.current);
      streamFlushRef.current = null;
    }
    const content = pendingAnswerRef.current;
    pendingAnswerRef.current = "";
    if (content) {
      useResearchStore.getState().handleEvent({
        type: "answer_chunk",
        content,
      });
    }
  }, []);

  // 组件卸载时清理 SSE 连接和超时
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (streamFlushRef.current) clearTimeout(streamFlushRef.current);
      pendingAnswerRef.current = "";
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
      if (streamFlushRef.current) {
        clearTimeout(streamFlushRef.current);
        streamFlushRef.current = null;
      }
      pendingAnswerRef.current = "";

      // 使用 getState 确保拿到最新 setState action，避免闭包陈旧引用
      useResearchStore.setState({
        status: "streaming", error: null, plan: null, subtasks: {},
        synthesizing: false, criticScore: null, refineRound: 0,
        finalResult: null, streamingAnswer: "",
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
          if (event.type === "answer_chunk") {
            const currentLength =
              useResearchStore.getState().streamingAnswer.length;
            const remaining = Math.max(
              0,
              MAX_STREAMED_ANSWER_CHARS -
                currentLength -
                pendingAnswerRef.current.length,
            );
            if (remaining > 0) {
              pendingAnswerRef.current += event.content.slice(0, remaining);
            }
            if (!streamFlushRef.current && pendingAnswerRef.current) {
              streamFlushRef.current = setTimeout(
                flushPendingAnswer,
                STREAM_FLUSH_MS,
              );
            }
            return;
          }
          flushPendingAnswer();
          store.handleEvent(event);
          if (event.type === "error") {
            clearTimeout(timeoutId);
            abortRef.current?.abort();
            return;
          }
          if (event.type === "done") {
            clearTimeout(timeoutId);
            if (!event.result.success) return;
            const result = event.result as unknown as Record<string, unknown> | undefined;
            const report = (result?.output as string) || "";
            const quality = (result?.metadata as Record<string, unknown> | undefined)?.quality as number | undefined;
            const model = (result?.metadata as Record<string, unknown> | undefined)?.model as string | undefined;
            void addFromResearch(task, report, quality, model);
          }
        },
        () => {
          clearTimeout(timeoutId);
          // 仅当 done 事件已将 finalResult 写入后才置 completed，
          // 避免 [DONE] 标记先于 done 事件到达时出现"已完成但无报告"白屏
          const current = useResearchStore.getState();
          if (current.finalResult?.success) {
            useResearchStore.getState().setStatus("completed");
          } else if (current.status !== "error") {
            useResearchStore.getState().setStatus(
              "error",
              "研究连接已结束，但未收到完整结果",
            );
          }
        },
        (err) => {
          clearTimeout(timeoutId);
          const msg = err.message || "";
          // 分类错误信息，提供用户友好的中文提示
          if (err instanceof Error && "status" in err) {
            const status = (err as unknown as Record<string, unknown>).status as number;
            if (status === 401 || status === 403) {
              store.setStatus("error", "API Key 无效或已过期，请在设置中更新 DeepSeek Key。");
              return;
            }
            if (status >= 500) {
              store.setStatus("error", "服务器繁忙，请稍后重试。若持续出现请检查 API Key 余额。");
              return;
            }
          }
          const lower = msg.toLowerCase();
          if (lower.includes("401") || lower.includes("403") || lower.includes("auth")) {
            store.setStatus("error", "API Key 无效或已过期，请在设置中更新 DeepSeek Key。");
          } else if (lower.includes("timeout") || lower.includes("abort")) {
            store.setStatus("error", "研究超时，请尝试简化问题或减少问题范围。");
          } else if (lower.includes("network") || lower.includes("fetch") || lower.includes("connect")) {
            store.setStatus("error", "网络连接失败，请检查网络后重试。");
          } else if (msg && msg.length < 80) {
            // 简短的后端消息，可能是中文错误，直接展示
            store.setStatus("error", msg);
          } else {
            // 长错误/未知错误，给通用提示
            store.setStatus("error", "研究请求失败，请稍后重试。如持续出现请检查 API Key 余额。");
          }
        },
      );
    },
    [store, addFromResearch, flushPendingAnswer],
  );

  const cancelResearch = useCallback(() => {
    abortRef.current?.abort();
    if (streamFlushRef.current) {
      clearTimeout(streamFlushRef.current);
      streamFlushRef.current = null;
    }
    pendingAnswerRef.current = "";
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
