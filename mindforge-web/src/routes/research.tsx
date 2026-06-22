import { createRoute } from "@tanstack/react-router";
import { ResearchPage } from "@/components/pages/research-page";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/research",
  component: ResearchPage,
});
