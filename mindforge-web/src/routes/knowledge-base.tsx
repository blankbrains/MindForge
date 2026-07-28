import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { importWithReload } from "@/lib/lazy-import";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/knowledge-base",
  component: lazyRouteComponent(
    () => importWithReload(
      "knowledge-base-page",
      () => import("@/components/pages/knowledge-base-page"),
    ),
    "KnowledgeBasePage",
  ),
});
