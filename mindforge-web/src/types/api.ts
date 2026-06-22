export interface HealthResponse {
  status: string;
  version: string;
  qdrant_connected: boolean;
  redis_connected: boolean;
  mcp_tools_available: boolean;
}

export interface StatsResponse {
  documents_indexed: number;
  qdrant_url: string;
  redis_url: string;
}
