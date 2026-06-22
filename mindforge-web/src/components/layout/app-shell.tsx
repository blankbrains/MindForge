import { Outlet, Link } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { useIsMobile } from "@/hooks/use-media-query";

export function AppShell() {
  const isMobile = useIsMobile();

  return (
    <div className="flex min-h-screen">
      {!isMobile && <Sidebar />}

      <div className="flex flex-1 flex-col xl:ml-60">
        <Header title="MindForge" />

        <main className="flex-1 overflow-y-auto p-6 pb-20 xl:pb-6">
          <Outlet />
        </main>
      </div>

      {/* Mobile bottom navigation */}
      {isMobile && (
        <nav className="fixed bottom-0 left-0 right-0 z-40 flex items-center justify-around border-t border-border bg-surface/95 pb-safe py-2 backdrop-blur">
          {[
            { label: "概览", to: "/" as const },
            { label: "研究", to: "/research" as const },
            { label: "知识库", to: "/knowledge-base" as const },
            { label: "设置", to: "/settings" as const },
          ].map(({ label, to }) => (
            <Link
              key={to}
              to={to}
              search={{}}
              className="flex flex-col items-center gap-0.5 px-3 py-1 text-xs text-text-muted [&.active]:text-primary"
            >
              {label}
            </Link>
          ))}
        </nav>
      )}
    </div>
  );
}
