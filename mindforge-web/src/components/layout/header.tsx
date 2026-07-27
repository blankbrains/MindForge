import { Menu, Sun, Moon } from "lucide-react";
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
      <button
        type="button"
        onClick={toggleSidebar}
        aria-label="切换侧边栏"
        className="rounded-lg p-2 text-text-muted hover:bg-surface-alt md:hidden"
      >
        <Menu className="h-5 w-5" aria-hidden="true" />
      </button>

      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>

      <div className="flex-1" />

      {/* Theme toggle */}
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

      {/* Connection indicator — driven by /health polling */}
      <div
        className={
          connectionState === "online"
            ? "flex items-center gap-1.5 rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700 dark:bg-green-900/40 dark:text-green-300"
            : connectionState === "degraded"
              ? "flex items-center gap-1.5 rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
              : "flex items-center gap-1.5 rounded-full bg-surface-alt px-3 py-1 text-xs font-medium text-text-muted"
        }
        aria-live="polite"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
        {connectionState === "online"
          ? "在线"
          : connectionState === "degraded"
            ? "部分服务异常"
            : "检测中"}
      </div>
    </header>
  );
}

export { Header };
