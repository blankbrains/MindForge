import { Outlet } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { useIsMobile } from "@/hooks/use-media-query";

export function AppShell() {
  const isMobile = useIsMobile();

  return (
    <div className="flex min-h-screen">
      {/* Sidebar — hidden on mobile */}
      {!isMobile && <Sidebar />}

      <div className="flex flex-1 flex-col xl:ml-60">
        <Header title="MindForge" />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>

      {/* Mobile bottom nav */}
      {isMobile && (
        <nav className="fixed bottom-0 left-0 right-0 z-40 flex items-center justify-around border-t border-border bg-surface py-2">
          {[
            { label: "概览", href: "/" },
            { label: "研究", href: "/research" },
            { label: "知识库", href: "/knowledge-base" },
            { label: "历史", href: "/history" },
          ].map(({ label, href }) => (
            <a
              key={href}
              href={href}
              className="flex flex-col items-center gap-0.5 px-3 py-1 text-xs text-text-muted"
            >
              {label}
            </a>
          ))}
        </nav>
      )}
    </div>
  );
}
