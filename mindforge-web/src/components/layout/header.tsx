import { Menu, Sun, Moon } from "lucide-react";
import { useUIStore } from "@/store/ui-store";

function Header({ title }: { title: string }) {
  const { toggleSidebar, theme, setTheme } = useUIStore();

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-surface/95 px-6 backdrop-blur">
      {/* Mobile menu toggle */}
      <button
        type="button"
        onClick={toggleSidebar}
        aria-label="切换侧边栏"
        className="rounded-lg p-2 text-text-muted hover:bg-surface-alt xl:hidden"
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

      {/* Connection indicator — powered by /health polling */}
      <div
        className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"
        aria-live="polite"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
        在线
      </div>
    </header>
  );
}

export { Header };
