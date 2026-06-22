import { Menu, Sun, Moon } from "lucide-react";
import { useUIStore } from "@/store/ui-store";
import { cn } from "@/lib/utils";

interface HeaderProps {
  title: string;
}

export function Header({ title }: HeaderProps) {
  const { sidebarOpen, toggleSidebar, theme, setTheme } = useUIStore();

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-surface/95 px-6 backdrop-blur">
      {/* Mobile menu toggle */}
      <button
        type="button"
        onClick={toggleSidebar}
        className="rounded-lg p-2 text-text-muted hover:bg-surface-alt xl:hidden"
      >
        <Menu className="h-5 w-5" />
      </button>

      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>

      <div className="flex-1" />

      {/* Theme toggle */}
      <button
        type="button"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="rounded-lg p-2 text-text-muted hover:bg-surface-alt"
        title={theme === "dark" ? "切换亮色模式" : "切换暗色模式"}
      >
        {theme === "dark" ? (
          <Sun className="h-5 w-5" />
        ) : (
          <Moon className="h-5 w-5" />
        )}
      </button>

      {/* Connection indicator — simplified */}
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium",
          "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
        )}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
        在线
      </div>
    </header>
  );
}
