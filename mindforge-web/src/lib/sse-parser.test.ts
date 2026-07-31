import { describe, expect, it, vi } from "vitest";

import { createSSEConnection, SSEConnectionError } from "@/lib/sse-parser";

describe("createSSEConnection", () => {
  it("preserves the HTTP status for failed connections", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"detail":"invalid key"}', {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const onError = vi.fn();

    createSSEConnection(
      "/api/v1/query",
      { task: "test" },
      vi.fn(),
      vi.fn(),
      onError,
    );

    await vi.waitFor(() => expect(onError).toHaveBeenCalledOnce());
    const error = onError.mock.calls[0][0];
    expect(error).toBeInstanceOf(SSEConnectionError);
    expect(error.status).toBe(401);
  });

  it("cancels the active stream reader when aborted", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    let finishRead!: (value: ReadableStreamReadResult<Uint8Array>) => void;
    const pendingRead = new Promise<ReadableStreamReadResult<Uint8Array>>(
      (resolve) => {
        finishRead = resolve;
      },
    );
    const read = vi.fn(() => pendingRead);
    const releaseLock = vi.fn();
    const onComplete = vi.fn();
    const onError = vi.fn();
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      body: {
        getReader: () => ({ cancel, read, releaseLock }),
      },
    } as unknown as Response);

    const connection = createSSEConnection(
      "/api/v1/query",
      { task: "cancel test" },
      vi.fn(),
      onComplete,
      onError,
    );
    await vi.waitFor(() => expect(read).toHaveBeenCalledOnce());

    connection.abort();
    connection.abort();
    finishRead({ done: true, value: undefined });

    await vi.waitFor(() => expect(cancel).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(releaseLock).toHaveBeenCalledOnce());
    expect(onComplete).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
