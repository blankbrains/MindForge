import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center py-20 text-center",
        className,
      )}
    >
      {icon && (
        <div className="mb-4 text-text-muted opacity-40">{icon}</div>
      )}
      <h3 className="text-lg font-medium text-text">{title}</h3>
      {description && (
        <p className="mt-2 max-w-sm text-sm text-text-muted">
          {description}
        </p>
      )}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}
