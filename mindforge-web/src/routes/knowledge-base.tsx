import { createRoute } from "@tanstack/react-router";
import { KnowledgeBasePage } from "@/components/pages/knowledge-base-page";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/knowledge-base",
  component: KnowledgeBasePage,
});
