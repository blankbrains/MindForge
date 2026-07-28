import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { importWithReload } from "@/lib/lazy-import";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: lazyRouteComponent(
    () => importWithReload(
      "dashboard-page",
      () => import("@/components/pages/dashboard-page"),
    ),
    "DashboardPage",
  ),
});
