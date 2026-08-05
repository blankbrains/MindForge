import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { API_BASE } from "@/lib/constants";
import { normalizeCitationSources } from "@/lib/citations";
import { createSSEConnection } from "@/lib/sse-parser";
import { useHistoryStore } from "@/store/history-store";
import { useResearchStore } from "@/store/research-store";
import { useSettingsStore } from "@/store/settings-store";
import type { SSEEvent } from "@/types/research";

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
  Number.isFinite(configuredMaxStreamedAnswerChars)
  && configuredMaxStreamedAnswerChars > 0
    ? configuredMaxStreamedAnswerChars
    : 1_000_000;

interface ResearchConnection {
  abort: () => void;
}

export interface ResearchContextOptions {
  conversationId?: string | null;
  contextMode?: "auto" | "manual" | "disabled";
  selectedContextIds?: string[];
  excludedContextIds?: string[];
  independent?: boolean;
}

let nextRequestId = 0;
let activeRequestId: number | null = null;
let activeServerRequestId: string | null = null;
let activeConnection: ResearchConnection | null = null;
let researchTimeout: ReturnType<typeof setTimeout> | null = null;
let streamFlushTimeout: ReturnType<typeof setTimeout> | null = null;
let pendingAnswer = "";

function createServerRequestId(requestId: number): string {
  return [
    "research",
    Date.now().toString(36),
    requestId.toString(36),
    Math.random().toString(36).slice(2, 14),
  ].join("-");
}

async function cancelServerResearch(requestId: string): Promise<void> {
  if (typeof fetch !== "function") return;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 1_500);
  try {
    await fetch(`${API_BASE}/query/cancel`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ request_id: requestId }),
      keepalive: true,
      signal: controller.signal,
    });
  } catch {
    // Disconnecting the SSE stream remains the fallback cancellation signal.
  } finally {
    clearTimeout(timeoutId);
  }
}

function stopResearchTransport(
  requestId: string | null,
  connection: ResearchConnection | null,
): void {
  if (!requestId) {
    connection?.abort();
    return;
  }
  void cancelServerResearch(requestId).finally(() => connection?.abort());
}

function isRunningStatus(status: string): boolean {
  return status === "connecting" || status === "streaming";
}

function clearResearchTimeout(): void {
  if (!researchTimeout) return;
  clearTimeout(researchTimeout);
  researchTimeout = null;
}

function clearPendingAnswer(): void {
  if (streamFlushTimeout) {
    clearTimeout(streamFlushTimeout);
    streamFlushTimeout = null;
  }
  pendingAnswer = "";
}

function flushPendingAnswer(): void {
  if (streamFlushTimeout) {
    clearTimeout(streamFlushTimeout);
    streamFlushTimeout = null;
  }
  const content = pendingAnswer;
  pendingAnswer = "";
  if (content) {
    useResearchStore.getState().handleEvent({
      type: "answer_chunk",
      content,
    });
  }
}

function detachActiveTransport(cancelServer: boolean): void {
  activeRequestId = null;
  const serverRequestId = activeServerRequestId;
  activeServerRequestId = null;
  const connection = activeConnection;
  activeConnection = null;
  clearResearchTimeout();
  clearPendingAnswer();
  if (cancelServer) {
    stopResearchTransport(serverRequestId, connection);
  } else {
    connection?.abort();
  }
}

function persistResearchResult(
  task: string,
  event: Extract<SSEEvent, { type: "done" }>,
): void {
  if (!event.result.success) return;
  const result = event.result as unknown as
    | Record<string, unknown>
    | undefined;
  const report = (result?.output as string) || "";
  const rawQuality = (
    result?.metadata as Record<string, unknown> | undefined
  )?.quality;
  const quality =
    typeof rawQuality === "number" && Number.isFinite(rawQuality)
      ? rawQuality
      : undefined;
  const model = (
    result?.metadata as Record<string, unknown> | undefined
  )?.model as string | undefined;
  const tokenUsage = result?.token_usage as
    | Record<string, number>
    | undefined;
  const costUsd = result?.cost_usd as number | null | undefined;
  const costStatus = result?.cost_status as string | undefined;
  const resultData = result?.data as Record<string, unknown> | undefined;
  const sources = normalizeCitationSources(resultData?.sources);
  const traceId =
    (result?.trace_id as string | undefined)
    ?? event.trace_id
    ?? useResearchStore.getState().traceId
    ?? undefined;
  const metadata = result?.metadata as Record<string, unknown> | undefined;
  const conversationId =
    typeof metadata?.conversation_id === "string"
      ? metadata.conversation_id
      : undefined;
  const runId =
    typeof metadata?.run_id === "string" ? metadata.run_id : undefined;
  const usageSummary = {
    tokenUsage,
    costUsd,
    costStatus,
  };
  const addFromResearch = useHistoryStore.getState().addFromResearch;
  if (traceId) {
    if (conversationId || runId) {
      void addFromResearch(
        task,
        report,
        quality,
        model,
        usageSummary,
        sources,
        traceId,
        conversationId,
        runId,
      );
      return;
    }
    void addFromResearch(
      task,
      report,
      quality,
      model,
      usageSummary,
      sources,
      traceId,
    );
    return;
  }
  void addFromResearch(
    task,
    report,
    quality,
    model,
    usageSummary,
    sources,
  );
}

export function startResearch(
  task: string,
  options: ResearchContextOptions = {},
): void {
  detachActiveTransport(true);

  const requestId = ++nextRequestId;
  const serverRequestId = createServerRequestId(requestId);
  activeRequestId = requestId;
  activeServerRequestId = serverRequestId;

  useResearchStore.setState({
    status: "streaming",
    error: null,
    plan: null,
    subtasks: {},
    planning: false,
    synthesizing: false,
    criticScore: null,
    refineRound: 0,
    finalResult: null,
    streamingAnswer: "",
    traceId: null,
    activeTask: task,
    phase: "connecting",
    startedAt: Date.now(),
    lastHeartbeatAt: null,
  });
  useResearchStore.getState().setTask(task);

  const settingsState = useSettingsStore.getState();
  const configuredSeconds = settingsState.researchTimeout;
  const providerLabel =
    settingsState.providerConfigs[settingsState.llmProvider]?.label
    || "当前模型服务";
  const authenticationError =
    `API Key 无效或已过期，请在设置中更新${providerLabel}凭证。`;
  const timeoutMs =
    Number.isFinite(configuredSeconds) && configuredSeconds > 0
      ? configuredSeconds * 1000
      : RESEARCH_TIMEOUT_MS;

  researchTimeout = setTimeout(() => {
    if (activeRequestId !== requestId) return;
    const connection = activeConnection;
    activeConnection = null;
    activeRequestId = null;
    if (activeServerRequestId === serverRequestId) {
      activeServerRequestId = null;
    }
    researchTimeout = null;
    clearPendingAnswer();
    useResearchStore
      .getState()
      .fail(
        `研究超时（${Math.ceil(timeoutMs / 60_000)} 分钟），请尝试简化问题`,
      );
    stopResearchTransport(serverRequestId, connection);
  }, timeoutMs);

  activeConnection = createSSEConnection<SSEEvent>(
    `${API_BASE}/query`,
    {
      request_id: serverRequestId,
      task,
      stream: true,
      conversation_id: options.conversationId ?? undefined,
      context_mode: options.contextMode ?? undefined,
      selected_context_ids: options.selectedContextIds ?? [],
      excluded_context_ids: options.excludedContextIds ?? [],
      independent: options.independent ?? false,
    },
    (event) => {
      if (activeRequestId !== requestId) return;
      if (event.type === "answer_chunk") {
        const currentLength =
          useResearchStore.getState().streamingAnswer.length;
        const remaining = Math.max(
          0,
          MAX_STREAMED_ANSWER_CHARS
            - currentLength
            - pendingAnswer.length,
        );
        if (remaining > 0) {
          pendingAnswer += event.content.slice(0, remaining);
        }
        if (!streamFlushTimeout && pendingAnswer) {
          streamFlushTimeout = setTimeout(
            flushPendingAnswer,
            STREAM_FLUSH_MS,
          );
        }
        return;
      }

      flushPendingAnswer();
      useResearchStore.getState().handleEvent(event);

      if (event.type === "error") {
        clearResearchTimeout();
        activeRequestId = null;
        if (activeServerRequestId === serverRequestId) {
          activeServerRequestId = null;
        }
        const connection = activeConnection;
        activeConnection = null;
        connection?.abort();
        return;
      }

      if (event.type === "done") {
        clearResearchTimeout();
        if (activeServerRequestId === serverRequestId) {
          activeServerRequestId = null;
        }
        persistResearchResult(task, event);
      }
    },
    () => {
      if (activeRequestId !== requestId) return;
      activeRequestId = null;
      if (activeServerRequestId === serverRequestId) {
        activeServerRequestId = null;
      }
      clearResearchTimeout();
      clearPendingAnswer();
      activeConnection = null;
      const current = useResearchStore.getState();
      if (!current.finalResult?.success && current.status !== "error") {
        current.fail("研究连接已结束，但未收到完整结果");
      }
    },
    (error) => {
      if (activeRequestId !== requestId) return;
      activeRequestId = null;
      if (activeServerRequestId === serverRequestId) {
        activeServerRequestId = null;
      }
      clearResearchTimeout();
      clearPendingAnswer();
      activeConnection = null;
      const message = error.message || "";
      if (error instanceof Error && "status" in error) {
        const status = (error as unknown as Record<string, unknown>)
          .status as number;
        if (status === 401 || status === 403) {
          useResearchStore.getState().fail(authenticationError);
          return;
        }
        if (status >= 500) {
          useResearchStore
            .getState()
            .fail(
              "服务器繁忙，请稍后重试。若持续出现请检查 API Key 余额。",
            );
          return;
        }
      }
      const lower = message.toLowerCase();
      if (
        lower.includes("401")
        || lower.includes("403")
        || lower.includes("auth")
      ) {
        useResearchStore.getState().fail(authenticationError);
      } else if (
        lower.includes("timeout")
        || lower.includes("abort")
      ) {
        useResearchStore
          .getState()
          .fail("研究超时，请尝试简化问题或减少问题范围。");
      } else if (
        lower.includes("network")
        || lower.includes("fetch")
        || lower.includes("connect")
      ) {
        useResearchStore
          .getState()
          .fail("网络连接失败，请检查网络后重试。");
      } else if (message && message.length < 80) {
        useResearchStore.getState().fail(message);
      } else {
        useResearchStore
          .getState()
          .fail(
            "研究请求失败，请稍后重试。如持续出现请检查 API Key 余额。",
          );
      }
    },
  );
}

export function cancelResearch(): void {
  const running = isRunningStatus(useResearchStore.getState().status);
  detachActiveTransport(true);
  if (running) {
    useResearchStore.getState().interrupt("cancelled");
  }
}

function cancelResearchOnUnload(): void {
  const serverRequestId = activeServerRequestId;
  const connection = activeConnection;
  if (
    activeRequestId === null
    && !serverRequestId
    && !connection
  ) {
    return;
  }

  activeRequestId = null;
  activeServerRequestId = null;
  activeConnection = null;
  clearResearchTimeout();
  clearPendingAnswer();
  connection?.abort();

  if (!serverRequestId || typeof fetch !== "function") return;
  void fetch(`${API_BASE}/query/cancel`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ request_id: serverRequestId }),
    keepalive: true,
  }).catch(() => {});
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", cancelResearchOnUnload);
}

export function useResearchSession() {
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

  useEffect(() => {
    const current = useResearchStore.getState();
    if (isRunningStatus(current.status) && activeRequestId === null) {
      current.interrupt();
    }
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
    isCancelled: sessionState.status === "cancelled",
    isError: sessionState.status === "error",
  };
}
