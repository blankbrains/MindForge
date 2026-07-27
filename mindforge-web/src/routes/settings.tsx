import {
  createRoute,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: lazyRouteComponent(
    () => import("@/components/pages/settings-page"),
    "SettingsPage",
  ),
});
