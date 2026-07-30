import { describe, expect, it } from "vitest";

import {
  formatCostEstimate,
  formatTokenCount,
} from "@/lib/utils";

describe("research metric formatting", () => {
  it("does not present unknown pricing as a zero-dollar cost", () => {
    expect(
      formatCostEstimate(null, "pricing_unconfigured"),
    ).toBe("未配置模型价格");
    expect(
      formatCostEstimate(null, "usage_unavailable"),
    ).toBe("API 未返回用量");
  });

  it("formats estimated and partial costs distinctly", () => {
    expect(formatCostEstimate(0.001, "estimated")).toBe("$0.001000");
    expect(formatCostEstimate(0.001, "partial")).toContain("部分估算");
  });

  it("formats token totals for scanning", () => {
    expect(formatTokenCount(12345)).toBe("12,345");
  });
});
