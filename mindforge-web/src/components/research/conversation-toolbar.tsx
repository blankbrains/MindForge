import {
  Eye,
  MessageSquarePlus,
  Trash2,
} from "lucide-react";
import type { ContextMode, Conversation } from "@/types/context";

interface ConversationToolbarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  contextMode: ContextMode;
  independent: boolean;
  disabled: boolean;
  loading: boolean;
  onSelect: (conversationId: string) => void;
  onCreate: () => void;
  onDelete: () => void;
  onModeChange: (mode: ContextMode) => void;
  onIndependentChange: (independent: boolean) => void;
  onOpenContext: () => void;
}

const MODES: Array<{ value: ContextMode; label: string }> = [
  { value: "auto", label: "自动" },
  { value: "manual", label: "手动" },
  { value: "disabled", label: "关闭" },
];

export function ConversationToolbar({
  conversations,
  activeConversationId,
  contextMode,
  independent,
  disabled,
  loading,
  onSelect,
  onCreate,
  onDelete,
  onModeChange,
  onIndependentChange,
  onOpenContext,
}: ConversationToolbarProps) {
  return (
    <section
      aria-label="会话与上下文控制"
      className="flex flex-col gap-3 border-y border-border py-3 lg:flex-row lg:items-center lg:justify-between"
    >
      <div className="flex min-w-0 items-center gap-2">
        <label htmlFor="conversation-select" className="sr-only">
          当前研究会话
        </label>
        <select
          id="conversation-select"
          value={activeConversationId ?? ""}
          disabled={disabled || loading}
          onChange={(event) => onSelect(event.target.value)}
          className="min-w-0 flex-1 rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 lg:w-64 lg:flex-none"
        >
          {conversations.map((conversation) => (
            <option
              key={conversation.conversation_id}
              value={conversation.conversation_id}
            >
              {conversation.title}
            </option>
          ))}
        </select>
        <button
          type="button"
          title="新建会话"
          aria-label="新建会话"
          disabled={disabled || loading}
          onClick={onCreate}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border bg-surface text-text-muted hover:bg-surface-alt hover:text-text disabled:opacity-50"
        >
          <MessageSquarePlus className="h-4 w-4" />
        </button>
        <button
          type="button"
          title="彻底删除当前会话"
          aria-label="彻底删除当前会话"
          disabled={disabled || loading || !activeConversationId}
          onClick={onDelete}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border bg-surface text-text-muted hover:border-red-300 hover:bg-red-50 hover:text-red-600 disabled:opacity-50 dark:hover:bg-red-950"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div
          role="group"
          aria-label="上下文模式"
          className="inline-flex rounded-md border border-border bg-surface p-0.5"
        >
          {MODES.map((mode) => (
            <button
              key={mode.value}
              type="button"
              disabled={disabled || independent}
              aria-pressed={contextMode === mode.value}
              onClick={() => onModeChange(mode.value)}
              className={`min-w-14 rounded px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-45 ${
                contextMode === mode.value
                  ? "bg-primary text-white"
                  : "text-text-muted hover:bg-surface-alt hover:text-text"
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        <label className="inline-flex items-center gap-2 text-sm text-text-muted">
          <input
            type="checkbox"
            checked={independent}
            disabled={disabled}
            onChange={(event) => onIndependentChange(event.target.checked)}
            className="h-4 w-4 accent-primary"
          />
          独立研究
        </label>

        <button
          type="button"
          disabled={disabled || !activeConversationId}
          onClick={onOpenContext}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface-alt disabled:opacity-50"
        >
          <Eye className="h-4 w-4" />
          查看上下文
        </button>
      </div>
    </section>
  );
}
