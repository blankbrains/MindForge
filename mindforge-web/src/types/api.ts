export interface HealthResponse {
  status: string;
  version: string;
  qdrant_connected: boolean;
  redis_connected: boolean;
  postgres_connected: boolean;
}

export interface StatsResponse {
  documents_indexed: number;
  chunks_indexed: number;
  qdrant_connected: boolean;
  qdrant_url: string;
  redis_url: string;
  max_upload_mb: number;
  max_pdf_pages: number;
  embedding_provider: string;
  embedding_device: string;
}
