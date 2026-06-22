import { Route as rootRoute } from "./routes/__root";
import { Route as IndexRoute } from "./routes/index";
import { Route as ResearchRoute } from "./routes/research";
import { Route as KnowledgeBaseRoute } from "./routes/knowledge-base";
import { Route as HistoryRoute } from "./routes/history";
import { Route as SettingsRoute } from "./routes/settings";

export const routeTree = rootRoute.addChildren([
  IndexRoute,
  ResearchRoute,
  KnowledgeBaseRoute,
  HistoryRoute,
  SettingsRoute,
]);
