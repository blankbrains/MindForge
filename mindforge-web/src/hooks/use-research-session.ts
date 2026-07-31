import { useCallback, useEffect, useRef } from "react";
import { API_BASE } from "@/lib/constants";
import type { SSEEvent } from "@/types/research";
import { useResearchStore } from "@/store/research-store";
import { useHistoryStore } from "@/store/history-store";
import { useSettingsStore } from "@/store/settings-store";
import { createSSEConnection } from "@/lib/sse-parser";
import { normalizeCitationSources } from "@/lib/citations";
import { useShallow } from "zustand/react/shallow";

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
  const requestGenerationRef = useRef(0);
  const sessionState = useResearchStore(
    useShallow((state) => ({
      status: state.status,
      error: state.error,
      plan: state.plan,
      subtasks: state.subtasks,
      planning: state.planning,
      synthesizing: state.synthesizing,
      criticScore: state.criticScore,
      refineRound: state.refineRound,
      finalResult: state.finalResult,
      traceId: state.traceId,
      phase: state.phase,
      startedAt: state.startedAt,
      lastHeartbeatAt: state.lastHeartbeatAt,
    })),
  );
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
      requestGenerationRef.current += 1;
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
      const generation = requestGenerationRef.current + 1;
      requestGenerationRef.current = generation;
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
        planning: false, synthesizing: false, criticScore: null, refineRound: 0,
        finalResult: null, streamingAnswer: "", traceId: null,
        activeTask: task, phase: "connecting", startedAt: Date.now(),
        lastHeartbeatAt: null,
      });
      useResearchStore.getState().setTask(task);

      const configuredSeconds =
        useSettingsStore.getState().researchTimeout;
      const settingsState = useSettingsStore.getState();
      const providerLabel =
        settingsState.providerConfigs[settingsState.llmProvider]?.label
        || "当前模型服务";
      const authenticationError =
        `API Key 无效或已过期，请在设置中更新${providerLabel}凭证。`;
      const timeoutMs =
        Number.isFinite(configuredSeconds) && configuredSeconds > 0
          ? configuredSeconds * 1000
          : RESEARCH_TIMEOUT_MS;
      const timeoutId = setTimeout(() => {
        if (requestGenerationRef.current !== generation) return;
        researchTimeoutRef.current = null;
        useResearchStore.getState().setStatus(
          "error",
          `研究超时（${Math.ceil(timeoutMs / 60_000)} 分钟），请尝试简化问题`,
        );
        abortRef.current?.abort();
      }, timeoutMs);
      researchTimeoutRef.current = timeoutId;

      abortRef.current = createSSEConnection<SSEEvent>(
        `${API_BASE}/query`,
        { task, stream: true },
        (event) => {
          if (requestGenerationRef.current !== generation) return;
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
          useResearchStore.getState().handleEvent(event);
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
            const tokenUsage = result?.token_usage as Record<string, number> | undefined;
            const costUsd = result?.cost_usd as number | null | undefined;
            const costStatus = result?.cost_status as string | undefined;
            const resultData = result?.data as
              | Record<string, unknown>
              | undefined;
            const sources = normalizeCitationSources(resultData?.sources);
            const traceId =
              (result?.trace_id as string | undefined)
              ?? event.trace_id
              ?? useResearchStore.getState().traceId
              ?? undefined;
            const usageSummary = {
              tokenUsage,
              costUsd,
              costStatus,
            };
            if (traceId) {
              void addFromResearch(
                task,
                report,
                quality,
                model,
                usageSummary,
                sources,
                traceId,
              );
            } else {
              void addFromResearch(
                task,
                report,
                quality,
                model,
                usageSummary,
                sources,
              );
            }
          }
        },
        () => {
          if (requestGenerationRef.current !== generation) return;
          clearTimeout(timeoutId);
          researchTimeoutRef.current = null;
          abortRef.current = null;
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
          if (requestGenerationRef.current !== generation) return;
          clearTimeout(timeoutId);
          researchTimeoutRef.current = null;
          abortRef.current = null;
          const msg = err.message || "";
          // 分类错误信息，提供用户友好的中文提示
          if (err instanceof Error && "status" in err) {
            const status = (err as unknown as Record<string, unknown>).status as number;
            if (status === 401 || status === 403) {
              useResearchStore.getState().setStatus("error", authenticationError);
              return;
            }
            if (status >= 500) {
              useResearchStore.getState().setStatus("error", "服务器繁忙，请稍后重试。若持续出现请检查 API Key 余额。");
              return;
            }
          }
          const lower = msg.toLowerCase();
          if (lower.includes("401") || lower.includes("403") || lower.includes("auth")) {
            useResearchStore.getState().setStatus("error", authenticationError);
          } else if (lower.includes("timeout") || lower.includes("abort")) {
            useResearchStore.getState().setStatus("error", "研究超时，请尝试简化问题或减少问题范围。");
          } else if (lower.includes("network") || lower.includes("fetch") || lower.includes("connect")) {
            useResearchStore.getState().setStatus("error", "网络连接失败，请检查网络后重试。");
          } else if (msg && msg.length < 80) {
            // 简短的后端消息，可能是中文错误，直接展示
            useResearchStore.getState().setStatus("error", msg);
          } else {
            // 长错误/未知错误，给通用提示
            useResearchStore.getState().setStatus("error", "研究请求失败，请稍后重试。如持续出现请检查 API Key 余额。");
          }
        },
      );
    },
    [addFromResearch, flushPendingAnswer],
  );

  const cancelResearch = useCallback(() => {
    requestGenerationRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    if (streamFlushRef.current) {
      clearTimeout(streamFlushRef.current);
      streamFlushRef.current = null;
    }
    pendingAnswerRef.current = "";
    if (researchTimeoutRef.current) { clearTimeout(researchTimeoutRef.current); researchTimeoutRef.current = null; }
    useResearchStore.getState().setStatus("idle");
    useResearchStore.setState({
      phase: "idle",
      activeTask: "",
      startedAt: null,
      lastHeartbeatAt: null,
    });
  }, []);

  return {
    ...sessionState,
    startResearch,
    cancelResearch,
    isIdle: sessionState.status === "idle",
    isStreaming:
      sessionState.status === "streaming"
      || sessionState.status === "connecting",
    isCompleted: sessionState.status === "completed",
    isError: sessionState.status === "error",
  };
}
