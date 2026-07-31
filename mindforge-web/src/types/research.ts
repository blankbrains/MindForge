export interface SubTask {
  task_id: string;
  description: string;
  task_type: "research" | "analysis" | "code" | "verify";
  dependencies: string[];
  status: "pending" | "in_progress" | "completed" | "failed";
  priority: number;
  result?: AgentResult;
}

export interface ResearchPlan {
  plan_id: string;
  original_task: string;
  subtasks: SubTask[];
  reasoning: string;
  planner_status?: "planned" | "direct" | "fallback";
  planner_error?: string | null;
}

export interface AgentResult {
  agent_name?: string;
  success: boolean;
  output: string;
  data?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  latency_ms?: number;
  cost_usd?: number | null;
  cost_status?:
    | "estimated"
    | "partial"
    | "pricing_unconfigured"
    | "usage_unavailable"
    | "not_applicable";
  token_usage?: Record<string, number>;
  trace_id?: string | null;
}

export type ResearchOutcome =
  | "success"
  | "degraded"
  | "retrieval_only"
  | "failed";

export interface CitationSource {
  index: number;
  title: string;
  url: string;
  source: string;
  chunk_id?: string;
  doc_id?: string;
}

// CriticScore — 与后端 critic.py CriticScore.to_dict() 对齐
export interface CriticScore {
  overall: number;
  completeness: number;
  accuracy: number;
  depth: number;
  clarity: number;
  citations: number;
  issues?: string[];
  suggestions?: string[];
  should_refine: boolean;
  evaluation_status?: "evaluated" | "failed";
  evaluation_error?: string | null;
}

type SSEEventPayload =
  | { type: "trace_started" }
  | { type: "planning"; status: "start" | "done" }
  | { type: "heartbeat"; timestamp: number }
  | { type: "plan_ready"; plan: ResearchPlan }
  | { type: "subtask_start"; task_id: string; description: string } // description 与 SubTask 冗余，由 SSE 协议定义决定
  | { type: "subtask_result"; task_id: string; result: AgentResult }
  | { type: "synthesizing"; status: "start" | "done" }
  | { type: "critic_feedback"; score: CriticScore; round: number }
  | { type: "refining"; round: number }
  | { type: "answer_chunk"; content: string }
  | {
      type: "error";
      content: string;
      stage?: string;
      code?: string;
      retryable?: boolean;
    }
  | { type: "done"; result: AgentResult };

export type SSEEvent = SSEEventPayload & { trace_id?: string | null };
