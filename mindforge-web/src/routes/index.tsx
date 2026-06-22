import { createRoute } from "@tanstack/react-router";
import { DashboardPage } from "@/components/pages/dashboard-page";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: DashboardPage,
});
