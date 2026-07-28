import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { importWithReload } from "@/lib/lazy-import";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/research",
  component: lazyRouteComponent(
    () => importWithReload(
      "research-page",
      () => import("@/components/pages/research-page"),
    ),
    "ResearchPage",
  ),
});
