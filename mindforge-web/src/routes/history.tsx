import { createRoute } from "@tanstack/react-router";
import { HistoryPage } from "@/components/pages/history-page";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history",
  component: HistoryPage,
});
