import { StatusCardsGrid } from "@/components/dashboard/status-cards-grid";
import { Link } from "@tanstack/react-router";
import { Search, Upload, Clock } from "lucide-react";

const quickActions = [
  { label: "新建研究", description: "提交一个新的研究问题", icon: Search, to: "/research", color: "bg-primary/10 text-primary" },
  { label: "上传文档", description: "向知识库添加文档", icon: Upload, to: "/knowledge-base", color: "bg-accent/10 text-accent" },
  { label: "查看历史", description: "浏览过去的研究记录", icon: Clock, to: "/history", color: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" },
];

export function DashboardPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">概览</h1>
        <p className="mt-1 text-text-muted">系统状态总览与快速操作入口</p>
      </div>
      <section>
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-text-muted">服务状态</h3>
        <StatusCardsGrid />
      </section>
      <section>
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-text-muted">快捷操作</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {quickActions.map((action) => (
            <Link key={action.to} to={action.to} className="group rounded-xl border border-border bg-surface p-6 transition-all hover:shadow-md hover:-translate-y-0.5">
              <div className={`inline-flex h-10 w-10 items-center justify-center rounded-lg ${action.color}`}>
                <action.icon className="h-5 w-5" />
              </div>
              <h4 className="mt-4 font-semibold">{action.label}</h4>
              <p className="mt-1 text-sm text-text-muted">{action.description}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
