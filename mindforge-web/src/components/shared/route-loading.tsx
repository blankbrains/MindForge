import { Loader2 } from "lucide-react";

export function RouteLoading() {
  return (
    <div
      className="flex min-h-64 items-center justify-center text-text-muted"
      role="status"
      aria-live="polite"
    >
      <Loader2
        className="h-6 w-6 animate-spin"
        aria-hidden="true"
      />
      <span className="ml-3 text-sm">正在加载页面</span>
    </div>
  );
}
