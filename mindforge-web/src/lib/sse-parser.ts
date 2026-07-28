import { createParser, type EventSourceMessage } from "eventsource-parser";

export type SSECallback<T> = (event: T) => void;

export class SSEConnectionError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "SSEConnectionError";
    this.status = status;
  }
}

function positiveEnvInt(name: string, fallback: number): number {
  const value = Number.parseInt(import.meta.env[name] || "", 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

const MAX_SSE_BYTES = positiveEnvInt("VITE_MAX_SSE_BYTES", 5 * 1024 * 1024);
const MAX_SSE_EVENT_CHARS = positiveEnvInt(
  "VITE_MAX_SSE_EVENT_CHARS",
  2_000_000,
);

export function createSSEConnection<T>(
  url: string,
  body: unknown,
  onEvent: SSECallback<T>,
  onComplete: () => void,
  onError: (err: Error) => void,
): { abort: () => void } {
  const controller = new AbortController();
  let completed = false; // 防重复触发 onComplete

  (async () => {
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new SSEConnectionError(
          detail || `SSE connection failed: ${response.status}`,
          response.status,
        );
      }
      if (!response.body) {
        throw new Error("Response has no body stream");
      }

      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let receivedBytes = 0;

      const parser = createParser({
        onEvent: (event: EventSourceMessage) => {
          if (event.data.length > MAX_SSE_EVENT_CHARS) {
            controller.abort();
            onError(new Error("SSE event exceeded the configured size limit"));
            return;
          }
          // 兼容尾部空白：trim 后比较
          if (!event.data || event.data.trim() === "[DONE]") {
            if (!completed) { completed = true; onComplete(); }
            reader?.cancel().catch(() => {});
            return;
          }
          try {
            const parsed = JSON.parse(event.data) as T;
            onEvent(parsed);
          } catch {
            // Skip unparseable events
          }
        },
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (!completed) { completed = true; onComplete(); }
          break;
        }
        receivedBytes += value.byteLength;
        if (receivedBytes > MAX_SSE_BYTES) {
          throw new Error("SSE response exceeded the configured size limit");
        }
        parser.feed(decoder.decode(value, { stream: !done }));
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    } finally {
      // 确保 reader 锁被释放
      try { reader?.releaseLock(); } catch { /* already released */ }
    }
  })();

  return {
    abort: () => {
      controller.abort();
    },
  };
}
