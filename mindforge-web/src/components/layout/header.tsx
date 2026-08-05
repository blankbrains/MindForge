import { Menu, Sun, Moon } from "lucide-react";
import { Tooltip } from "@/components/shared/tooltip";
import { useUIStore } from "@/store/ui-store";
import { useHealth } from "@/hooks/use-health";

function Header({ title }: { title: string }) {
  const { toggleSidebar, theme, setTheme } = useUIStore();
  const { data: health } = useHealth();
  const connectionState = !health
    ? "checking"
    : health.status === "ok"
      ? "online"
      : "degraded";

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-surface/95 px-6 backdrop-blur">
      {/* Mobile menu toggle */}
      <Tooltip content="打开或收起主导航" side="bottom">
        <button
          type="button"
          onClick={toggleSidebar}
          aria-label="切换侧边栏"
          className="rounded-lg p-2 text-text-muted hover:bg-surface-alt md:hidden"
        >
          <Menu className="h-5 w-5" aria-hidden="true" />
        </button>
      </Tooltip>

      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>

      <div className="flex-1" />

      {/* Theme toggle */}
      <Tooltip
        content={theme === "dark" ? "切换到亮色模式" : "切换到暗色模式"}
        side="bottom"
      >
        <button
          type="button"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={theme === "dark" ? "切换到亮色模式" : "切换到暗色模式"}
          className="rounded-lg p-2 text-text-muted hover:bg-surface-alt"
        >
          {theme === "dark" ? (
            <Sun className="h-5 w-5" aria-hidden="true" />
          ) : (
            <Moon className="h-5 w-5" aria-hidden="true" />
          )}
        </button>
      </Tooltip>

      {/* Connection indicator — driven by /health polling */}
      <Tooltip
        content={
          connectionState === "online"
            ? "API、PostgreSQL、Qdrant 和 Redis 均连接正常。"
            : connectionState === "degraded"
              ? "API 可访问，但至少一个数据库、向量库或缓存服务异常。"
              : "正在检查 API 与基础服务连接状态。"
        }
        side="bottom"
      >
        <div
          tabIndex={0}
          className={
            connectionState === "online"
              ? "flex cursor-help items-center gap-1.5 rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 outline-none focus-visible:ring-2 focus-visible:ring-primary/40 dark:bg-green-900/40 dark:text-green-300"
              : connectionState === "degraded"
                ? "flex cursor-help items-center gap-1.5 rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700 outline-none focus-visible:ring-2 focus-visible:ring-primary/40 dark:bg-amber-900/40 dark:text-amber-300"
                : "flex cursor-help items-center gap-1.5 rounded-full bg-surface-alt px-3 py-1 text-xs font-medium text-text-muted outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          }
          aria-live="polite"
        >
          <span
            className="h-1.5 w-1.5 rounded-full bg-current"
            aria-hidden="true"
          />
          {connectionState === "online"
            ? "在线"
            : connectionState === "degraded"
              ? "部分服务异常"
              : "检测中"}
        </div>
      </Tooltip>
    </header>
  );
}

export { Header };
