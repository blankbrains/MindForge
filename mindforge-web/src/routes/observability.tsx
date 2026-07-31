import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { importWithReload } from "@/lib/lazy-import";
import { Route as rootRoute } from "./__root";

interface ObservabilitySearch {
  traceId?: string;
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/observability",
  validateSearch: (search: Record<string, unknown>): ObservabilitySearch => ({
    traceId:
      typeof search.traceId === "string"
      && /^[0-9a-f]{32}$/.test(search.traceId)
        ? search.traceId
        : undefined,
  }),
  component: lazyRouteComponent(
    () =>
      importWithReload(
        "observability-page",
        () => import("@/components/pages/observability-page"),
      ),
    "ObservabilityPage",
  ),
});
