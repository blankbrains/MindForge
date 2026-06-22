import { createParser, type EventSourceMessage } from "eventsource-parser";

export type SSECallback<T> = (event: T) => void;

export function createSSEConnection<T>(
  url: string,
  body: unknown,
  onEvent: SSECallback<T>,
  onComplete: () => void,
  onError: (err: Error) => void,
): { abort: () => void } {
  const controller = new AbortController();

  (async () => {
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
        throw new Error(`SSE connection failed: ${response.status}`);
      }
      if (!response.body) {
        throw new Error("Response has no body stream");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const parser = createParser({
        onEvent: (event: EventSourceMessage) => {
          if (event.data === "[DONE]") {
            onComplete();
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
        if (done) break;
        parser.feed(decoder.decode(value, { stream: !done }));
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    }
  })();

  return {
    abort: () => {
      controller.abort();
    },
  };
}
