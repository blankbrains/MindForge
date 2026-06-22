import { rootRoute } from "./routes/__root";
import { Route as IndexRoute } from "./routes/index.lazy";
import { Route as ResearchRoute } from "./routes/research.lazy";
import { Route as KnowledgeBaseRoute } from "./routes/knowledge-base.lazy";
import { Route as HistoryRoute } from "./routes/history.lazy";
import { Route as SettingsRoute } from "./routes/settings.lazy";

export const routeTree = rootRoute.addChildren([
  IndexRoute,
  ResearchRoute,
  KnowledgeBaseRoute,
  HistoryRoute,
  SettingsRoute,
]);
