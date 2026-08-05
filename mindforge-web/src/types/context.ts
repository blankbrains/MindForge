export type ContextMode = "auto" | "manual" | "disabled";
export type ContextSourceType =
  | "message"
  | "summary"
  | "artifact"
  | "memory"
  | "document";

export interface Conversation {
  conversation_id: string;
  title: string;
  status: "active" | "archived";
  context_mode: ContextMode;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  message_id: string;
  conversation_id: string;
  run_id: string | null;
  role: "user" | "assistant" | "system_notice";
  content: string;
  sequence: number;
  include_in_context: boolean;
  pinned: boolean;
  context_scope: "turn" | "conversation" | "user";
  metadata: Record<string, unknown>;
  created_at: string;
  edited_at: string | null;
}

export interface ConversationDetail extends Conversation {
  messages: ConversationMessage[];
}

export interface ObservableContextItem {
  context_id: string;
  source_type: ContextSourceType;
  source_id: string;
  title: string;
  content: string;
  score: number;
  token_count: number;
  selection_reason: string;
  pinned: boolean;
  explicitly_selected: boolean;
  freshness_status: "current" | "stale" | "expired";
  included: boolean;
  exclusion_reason: string | null;
  metadata: Record<string, unknown>;
}

export interface ContextPreview {
  snapshot_id: string | null;
  standalone_query: string;
  requires_context: boolean;
  budget_tokens: number;
  used_tokens: number;
  context_fingerprint: string;
  policy_version: string;
  embedding_version: string;
  items: ObservableContextItem[];
  excluded: ObservableContextItem[];
}

export interface ContextSnapshot {
  snapshot_id: string;
  run_id: string;
  conversation_id: string | null;
  query_message_id: string | null;
  standalone_query: string;
  context_fingerprint: string;
  budget_tokens: number;
  used_tokens: number;
  policy_version: string;
  embedding_version: string;
  created_at: string;
  items: ObservableContextItem[];
}
