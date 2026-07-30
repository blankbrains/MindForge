import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatCost(usd: number): string {
  if (!Number.isFinite(usd)) return "—";
  if (usd < 0) return `-$${Math.abs(usd).toFixed(3)}`;
  if (usd < 0.01) return `$${usd.toFixed(6)}`;
  return `$${usd.toFixed(3)}`;
}

export function formatCostEstimate(
  usd: number | null | undefined,
  status?: string,
): string {
  if (status === "pricing_unconfigured") return "未配置模型价格";
  if (status === "usage_unavailable") return "API 未返回用量";
  if (status === "not_applicable") return "不涉及 API 费用";
  if (status === "partial") {
    return usd == null
      ? "部分调用无法估算"
      : `${formatCost(usd)}+（部分估算）`;
  }
  return usd == null ? "暂不可用" : formatCost(usd);
}

export function formatTokenCount(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens < 0) return "—";
  return Math.round(tokens).toLocaleString("zh-CN");
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
