import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/knowledge-base",
  component: lazyRouteComponent(
    () => import("@/components/pages/knowledge-base-page"),
    "KnowledgeBasePage",
  ),
});
