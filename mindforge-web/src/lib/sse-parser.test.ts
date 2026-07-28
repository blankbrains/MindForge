import { describe, expect, it, vi } from "vitest";

import {
  createSSEConnection,
  SSEConnectionError,
} from "@/lib/sse-parser";

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
});
