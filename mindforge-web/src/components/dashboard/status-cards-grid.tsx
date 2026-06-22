import { useHealth } from "@/hooks/use-health";
import { useStats } from "@/hooks/use-stats";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { Database, BarChart3, Wifi, Plug } from "lucide-react";

export function StatusCardsGrid() {
  const { data: health, isLoading: healthLoading } = useHealth();
  const { data: stats, isLoading: statsLoading } = useStats();

  const isLoading = healthLoading || statsLoading;

  const cards = [
    {
      label: "Qdrant 向量库",
      icon: Database,
      ok: health?.qdrant_connected ?? false,
    },
    {
      label: "Redis 缓存",
      icon: BarChart3,
      ok: health?.redis_connected ?? false,
    },
    {
      label: "MCP 协议",
      icon: Plug,
      ok: health?.mcp_tools_available ?? false,
    },
    {
      label: "已索引文档",
      icon: Wifi,
      ok: true,
      value: isLoading ? "-" : `${stats?.documents_indexed ?? 0}`,
      isCount: true,
      linkTo: "/knowledge-base" as const,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
      {cards.map(({ label, icon: Icon, ok, value, isCount, linkTo }) => {
        const inner = (
          <>
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-muted">{label}</span>
              <Icon className="h-4 w-4 text-text-muted opacity-60" />
            </div>
            <div className="mt-3 flex items-center gap-2">
              {isLoading ? (
                <div className="h-7 w-16 animate-pulse rounded bg-border/60" />
              ) : isCount ? (
                <span className="text-2xl font-bold">{value}</span>
              ) : (
                <>
                  <span
                    className={cn(
                      "h-2.5 w-2.5 rounded-full",
                      ok ? "bg-green-500" : "bg-red-500",
                    )}
                  />
                  <span
                    className={cn(
                      "text-lg font-semibold",
                      ok ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400",
                    )}
                  >
                    {ok ? "正常" : "离线"}
                  </span>
                </>
              )}
            </div>
          </>
        );

        if (linkTo) {
          return (
            <Link
              key={label}
              to={linkTo}
              search={{}}
              className="rounded-xl border border-border bg-surface p-5 transition-shadow hover:shadow-md hover:border-primary/30 block"
            >
              {inner}
            </Link>
          );
        }

        return (
          <div
            key={label}
            className="rounded-xl border border-border bg-surface p-5 transition-shadow hover:shadow-md"
          >
            {inner}
          </div>
        );
      })}
    </div>
  );
}
