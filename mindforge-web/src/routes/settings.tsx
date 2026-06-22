import { createRoute } from "@tanstack/react-router";
import { SettingsPage } from "@/components/pages/settings-page";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});
