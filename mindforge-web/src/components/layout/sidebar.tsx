import { Link, useLocation } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Search,
  Database,
  Clock,
  Settings,
  Zap,
} from "lucide-react";

const navItems = [
  { to: "/", label: "概览", icon: LayoutDashboard },
  { to: "/research", label: "研究", icon: Search },
  { to: "/knowledge-base", label: "知识库", icon: Database },
  { to: "/history", label: "历史", icon: Clock },
  { to: "/settings", label: "设置", icon: Settings },
];

export function Sidebar() {
  const loc = useLocation();
  const pathname = loc.pathname;

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-full w-60 flex-col border-r border-border bg-surface">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Zap className="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight">MindForge</h1>
          <p className="text-[10px] text-text-muted">自适应研究助理</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
        {navItems.map(({ to, label, icon: Icon }) => {
          const isActive =
            to === "/" ? pathname === "/" : pathname.startsWith(to);
          return (
            <Link
              key={to}
              to={to}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-text-muted hover:bg-surface-alt hover:text-text",
              )}
            >
              <Icon className="h-4.5 w-4.5" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border px-6 py-4">
        <p className="text-xs text-text-muted">MindForge v1.0.0</p>
      </div>
    </aside>
  );
}
