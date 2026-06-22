import { Link, Outlet, createRootRoute } from "@tanstack/react-router";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { Sun, Moon } from "lucide-react";
import { useUIStore } from "@/store/ui-store";

function RootLayout() {
  const { theme, setTheme } = useUIStore();

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex flex-1 flex-col xl:ml-60">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-surface/95 px-6 backdrop-blur">
          <h2 className="text-lg font-semibold tracking-tight">MindForge</h2>
          <div className="flex-1" />
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-lg p-2 text-text-muted hover:bg-surface-alt"
          >
            {theme === "dark" ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </button>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export const Route = createRootRoute({
  component: RootLayout,
});
