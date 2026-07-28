import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { importWithReload } from "@/lib/lazy-import";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history",
  component: lazyRouteComponent(
    () => importWithReload(
      "history-page",
      () => import("@/components/pages/history-page"),
    ),
    "HistoryPage",
  ),
});
