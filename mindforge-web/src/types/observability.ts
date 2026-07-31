export type TraceStatus =
  | "success"
  | "degraded"
  | "error"
  | "cancelled";

export interface ObservabilityStatus {
  enabled: boolean;
  local_storage: boolean;
  remote_configured: boolean;
  langfuse_host: string | null;
  capture_content: boolean;
  retention_days: number;
}

export interface TraceSummary {
  trace_id: string;
  name: string;
  display_name: string | null;
  start_time: number;
  end_time: number | null;
  duration_ms: number;
  status: TraceStatus;
  error: string | null;
  failure_summary: string | null;
  failure_count: number;
  task_preview: string | null;
  input?: unknown;
  output?: unknown;
  metadata: Record<string, unknown>;
  span_count: number;
  generation_count: number;
  tool_count: number;
  error_count: number;
  total_tokens: number;
  cost_usd: number | null;
  cost_status: string;
  remote_url: string | null;
}

export interface TraceListResponse {
  traces: TraceSummary[];
  total: number;
  limit: number;
  offset: number;
  truncated: boolean;
}

export interface TraceObservation {
  span_id: string;
  trace_id: string;
  name: string;
  start_time: number;
  end_time: number | null;
  duration_ms: number;
  parent_id: string | null;
  error: string | null;
  input?: unknown;
  output?: unknown;
  metadata: Record<string, unknown>;
  payloads_omitted?: string | null;
}

export interface TraceFailure {
  span_id: string;
  parent_id: string | null;
  observation_name: string;
  stage: string;
  error_code: string;
  error_type: string;
  message: string;
  status: string;
  agent: string | null;
  model: string | null;
  attempt: number | null;
}

export interface TraceDetailResponse {
  summary: TraceSummary;
  observations: TraceObservation[];
  failures: TraceFailure[];
  observations_truncated: boolean;
}
