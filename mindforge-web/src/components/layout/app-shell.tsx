import { Outlet, Link } from "@tanstack/react-router";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { useIsMobile } from "@/hooks/use-media-query";
import { useUIStore } from "@/store/ui-store";

export function AppShell() {
  const isMobile = useIsMobile();
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  return (
    <div className="flex min-h-screen">
      {/* Desktop: 固定侧边栏；Mobile: 抽屉式，受 sidebarOpen 控制 */}
      {isMobile ? (
        sidebarOpen && (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/50"
              onClick={toggleSidebar}
              aria-label="关闭侧边栏"
            />
            <div className="fixed left-0 top-0 z-50 h-full">
              <Sidebar />
            </div>
          </>
        )
      ) : (
        <>
          <Sidebar />
          {/* md+ 断点即可让位 Sidebar */}
          <div className="hidden md:block w-60 shrink-0" />
        </>
      )}

      <div className="flex flex-1 flex-col">
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
            { label: "历史", to: "/history" as const },
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
