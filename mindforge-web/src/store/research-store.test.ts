import { beforeEach, describe, expect, it } from "vitest";
import { useResearchStore } from "@/store/research-store";

describe("research store trace propagation", () => {
  beforeEach(() => {
    useResearchStore.getState().reset();
  });

  it("keeps the trace id from SSE through the final result", () => {
    const traceId = "a".repeat(32);
    useResearchStore.getState().handleEvent({
      type: "trace_started",
      trace_id: traceId,
    });
    useResearchStore.getState().handleEvent({
      type: "done",
      trace_id: traceId,
      result: {
        success: true,
        output: "report",
        trace_id: traceId,
      },
    });

    expect(useResearchStore.getState().traceId).toBe(traceId);
    expect(useResearchStore.getState().finalResult?.trace_id).toBe(traceId);
  });
});
