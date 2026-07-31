export interface DocumentItem {
  doc_id: string;
  filename: string;
  chunk_count: number;
  status: string;
  index_strategy: "auto" | "fixed" | "semantic";
  use_raptor: boolean;
  use_graphrag: boolean;
  source?: string;
  size_bytes?: number;
  indexed_at?: string;
}

export type IndexJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface IndexJob {
  job_id: string;
  doc_id?: string | null;
  filename: string;
  status: IndexJobStatus;
  stage: string;
  progress: number;
  chunk_count: number;
  timings: Record<string, number>;
  error?: string | null;
  cancel_requested: boolean;
  strategy: "auto" | "fixed" | "semantic";
  use_raptor: boolean;
  use_graphrag: boolean;
  created_at: string;
  updated_at: string;
}
