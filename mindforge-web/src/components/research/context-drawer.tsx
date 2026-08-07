import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  Ban,
  Check,
  Clock3,
  Loader2,
  Pin,
  PinOff,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { Tooltip } from "@/components/shared/tooltip";
import { useContextStore } from "@/store/context-store";
import type {
  ContextSourceType,
  ObservableContextItem,
} from "@/types/context";

interface ContextDrawerProps {
  open: boolean;
  task: string;
  onClose: () => void;
}

const SOURCE_LABELS: Record<ContextSourceType, string> = {
  message: "当前会话",
  summary: "会话摘要",
  artifact: "相关研究",
  memory: "长期记忆",
  document: "当前证据",
};
const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

function ContextRow({
  item,
  onToggle,
  onPin,
  onForget,
  onDelete,
  selectionDisabledReason,
  actionsDisabledReason,
  browseMode = false,
}: {
  item: ObservableContextItem;
  onToggle: (included: boolean) => void | Promise<void>;
  onPin: (pinned: boolean) => void | Promise<void>;
  onForget: () => void | Promise<void>;
  onDelete: () => void | Promise<void>;
  selectionDisabledReason?: string;
  actionsDisabledReason?: string;
  browseMode?: boolean;
}) {
  const mountedRef = useRef(true);
  const [pendingAction, setPendingAction] = useState<
    "pin" | "forget" | "delete" | null
  >(null);
  const sequence =
    typeof item.metadata.sequence === "number"
      ? item.metadata.sequence
      : null;
  const actionsDisabled = Boolean(actionsDisabledReason || pendingAction);
  const runAction = (
    action: "pin" | "forget" | "delete",
    callback: () => void | Promise<void>,
  ) => {
    if (actionsDisabled) return;
    setPendingAction(action);
    void Promise.resolve(callback()).finally(() => {
      if (mountedRef.current) {
        setPendingAction(null);
      }
    });
  };
  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );
  return (
    <li className="border-b border-border py-3 last:border-b-0">
      <div className="flex items-start gap-3">
        <Tooltip
          content={
            selectionDisabledReason
              ? selectionDisabledReason
              : browseMode
                ? item.included
                  ? "取消勾选后，后续研究不会使用这条会话消息。"
                  : "勾选后，后续研究可以使用这条会话消息。"
              : item.included
                ? "取消勾选后，该内容不会进入本轮研究上下文。"
                : "勾选后，该内容会加入本轮研究上下文。"
          }
          side="right"
          className="mt-1 shrink-0"
        >
          <input
            type="checkbox"
            checked={item.included}
            disabled={Boolean(selectionDisabledReason)}
            onChange={(event) => onToggle(event.target.checked)}
            aria-label={`${item.included ? "排除" : "包含"}${item.title}`}
            className="h-4 w-4 accent-primary"
          />
        </Tooltip>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="min-w-0 truncate text-sm font-semibold text-text">
              {item.title}
            </p>
            <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-text-muted">
              {SOURCE_LABELS[item.source_type]}
            </span>
            {item.source_type === "summary" && (
              <Tooltip
                content={
                  item.metadata.compression_method === "model"
                    ? `由 ${String(item.metadata.compression_model || "当前模型")} 对历史消息进行来源约束压缩。`
                    : `使用确定性规则压缩历史消息；模型压缩状态：${String(item.metadata.compression_status || "未执行")}。`
                }
                side="top"
              >
                <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-text-muted">
                  {item.metadata.compression_method === "model"
                    ? "模型压缩"
                    : "规则摘要"}
                </span>
              </Tooltip>
            )}
            {item.freshness_status !== "current" && (
              <span className="inline-flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-300">
                <Clock3 className="h-3 w-3" />
                {item.freshness_status === "expired" ? "已过期" : "需复核"}
              </span>
            )}
          </div>
          <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-xs leading-5 text-text-muted">
            {item.content}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-text-muted">
            {browseMode ? (
              <span>{sequence === null ? "会话消息" : `第 ${sequence} 条消息`}</span>
            ) : (
              <span>{item.token_count} tokens</span>
            )}
            <span>{item.selection_reason}</span>
            {!item.included && item.exclusion_reason && (
              <span className="text-amber-700 dark:text-amber-300">
                {item.exclusion_reason}
              </span>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {(item.source_type === "memory"
            || (
              item.source_type === "message"
              && item.metadata.context_eligible !== false
            )) && (
            <Tooltip
              content={
                actionsDisabledReason
                  ?? (
                    item.pinned
                      ? "取消固定后，该内容将重新按相关性和预算决定是否使用。"
                      : "固定后优先保留该内容，不会因相关性排序或预算不足被自动排除。"
                  )
              }
              side="left"
            >
              <button
                type="button"
                aria-label={item.pinned ? "取消固定" : "固定上下文"}
                aria-busy={pendingAction === "pin"}
                disabled={actionsDisabled}
                onClick={() =>
                  runAction("pin", () => onPin(!item.pinned))
                }
                className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-surface-alt hover:text-text disabled:cursor-not-allowed disabled:opacity-45"
              >
                {pendingAction === "pin" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : item.pinned ? (
                  <PinOff className="h-4 w-4" />
                ) : (
                  <Pin className="h-4 w-4" />
                )}
              </button>
            </Tooltip>
          )}
          {item.source_type === "message" && (
            <>
              {item.metadata.context_eligible !== false && (
                <Tooltip
                  content={
                    actionsDisabledReason
                    ?? "保留本轮问题与回答供你查看，但以后不再把它们用于上下文或记忆。"
                  }
                  side="left"
                >
                  <button
                    type="button"
                    aria-label="以后遗忘本轮问答"
                    aria-busy={pendingAction === "forget"}
                    disabled={actionsDisabled}
                    onClick={() => runAction("forget", onForget)}
                    className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-amber-50 hover:text-amber-700 disabled:cursor-not-allowed disabled:opacity-45 dark:hover:bg-amber-950"
                  >
                    {pendingAction === "forget" ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Ban className="h-4 w-4" />
                    )}
                  </button>
                </Tooltip>
              )}
              <Tooltip
                content={
                  actionsDisabledReason
                  ?? "永久删除本轮问题、回答，以及由它们生成的摘要、研究产物和长期记忆。"
                }
                side="left"
              >
                <button
                  type="button"
                  aria-label="彻底删除本轮问答"
                  aria-busy={pendingAction === "delete"}
                  disabled={actionsDisabled}
                  onClick={() => runAction("delete", onDelete)}
                  className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-45 dark:hover:bg-red-950"
                >
                  {pendingAction === "delete" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </button>
              </Tooltip>
            </>
          )}
        </div>
      </div>
    </li>
  );
}

export function ContextDrawer({ open, task, onClose }: ContextDrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(
    null,
  );
  const {
    contextMode,
    independent,
    activeConversation,
    selectedContextIds,
    excludedContextIds,
    preview,
    snapshot,
    previewLoading,
    error,
    toggleContextItem,
    previewContext,
    setPinned,
    forgetMessage,
    deleteMessage,
  } = useContextStore();
  const hasTask = Boolean(task.trim());
  const showingSnapshot =
    !hasTask
    && Boolean(snapshot)
    && selectedSnapshotId === snapshot?.snapshot_id;
  const browsingConversation = !hasTask && !showingSnapshot;
  const closeDrawer = useCallback(() => {
    setSelectedSnapshotId(null);
    onClose();
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    if (hasTask) {
      void previewContext(task);
    }
  }, [hasTask, open, previewContext, task]);

  useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const appRoot = document.getElementById("root");
    appRoot?.setAttribute("inert", "");
    appRoot?.setAttribute("aria-hidden", "true");
    panelRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeDrawer();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      );
      if (focusable.length === 0) {
        event.preventDefault();
        panelRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      appRoot?.removeAttribute("inert");
      appRoot?.removeAttribute("aria-hidden");
      previousFocus?.focus();
    };
  }, [closeDrawer, open]);

  const conversationItems = useMemo<ObservableContextItem[]>(() => {
    return (activeConversation?.messages ?? [])
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map((message) => {
        const contextId = `message:${message.message_id}`;
        const selected =
          selectedContextIds.includes(contextId)
          || selectedContextIds.includes(message.message_id);
        const excluded =
          excludedContextIds.includes(contextId)
          || excludedContextIds.includes(message.message_id);
        const included =
          message.include_in_context
          && (contextMode === "manual" ? selected : !excluded);
        let selectionReason = "可参与后续自动选择";
        if (!message.include_in_context) {
          selectionReason = "已从上下文中遗忘";
        } else if (contextMode === "manual") {
          selectionReason = selected ? "已手动选择" : "未手动选择";
        } else if (excluded) {
          selectionReason = "已从后续研究排除";
        }
        return {
          context_id: contextId,
          source_type: "message",
          source_id: message.message_id,
          title:
            message.role === "user" ? "用户问题" : "MindForge 回答",
          content: message.content,
          score: 0,
          token_count: 0,
          selection_reason: selectionReason,
          pinned: message.pinned,
          explicitly_selected: selected,
          freshness_status: "current",
          included,
          exclusion_reason: null,
          metadata: {
            ...message.metadata,
            role: message.role,
            sequence: message.sequence,
            context_eligible: message.include_in_context,
          },
        };
      });
  }, [
    activeConversation,
    contextMode,
    excludedContextIds,
    selectedContextIds,
  ]);
  const displayed = hasTask ? preview : (showingSnapshot ? snapshot : null);
  const grouped = useMemo(() => {
    const items = displayed?.items ?? [];
    const groups = items.reduce<
      Partial<Record<ContextSourceType, ObservableContextItem[]>>
    >((result, item) => {
      const group = result[item.source_type] ?? [];
      group.push(item);
      result[item.source_type] = group;
      return result;
    }, {});
    return Object.entries(groups) as Array<
      [ContextSourceType, ObservableContextItem[]]
    >;
  }, [displayed]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 bg-black/35"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closeDrawer();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="context-drawer-title"
        tabIndex={-1}
        className="ml-auto flex h-full w-full max-w-xl flex-col border-l border-border bg-surface shadow-2xl outline-none"
      >
        <header className="flex items-start justify-between border-b border-border px-5 py-4">
          <div>
            <h2 id="context-drawer-title" className="text-lg font-semibold">
              {showingSnapshot
                ? "本轮实际输入"
                : browsingConversation
                  ? "会话上下文"
                  : "运行前上下文预览"}
            </h2>
            <p className="mt-1 text-xs text-text-muted">
              {showingSnapshot
                ? "只读审计视图：展示本轮回答前实际送入模型的历史上下文。"
                : browsingConversation
                  ? "当前会话已完成的问答会在回答结束后立即显示在这里。"
                : independent
                ? "独立研究不会继承历史上下文"
                : `模式：${contextMode === "auto" ? "自动选择" : contextMode === "manual" ? "手动选择" : "关闭"}`}
            </p>
          </div>
          <Tooltip content="关闭上下文面板" side="left">
            <button
              type="button"
              aria-label="关闭上下文面板"
              onClick={closeDrawer}
              className="grid h-9 w-9 place-items-center rounded-md text-text-muted hover:bg-surface-alt hover:text-text"
            >
              <X className="h-5 w-5" />
            </button>
          </Tooltip>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {!hasTask && snapshot && (
            <div
              role="tablist"
              aria-label="上下文视图"
              className="mb-4 inline-flex rounded-md border border-border bg-surface-alt p-0.5"
            >
              <button
                type="button"
                role="tab"
                aria-selected={!showingSnapshot}
                onClick={() => setSelectedSnapshotId(null)}
                className={`rounded px-3 py-1.5 text-sm ${
                  !showingSnapshot
                    ? "bg-surface font-semibold text-text shadow-sm"
                    : "text-text-muted hover:text-text"
                }`}
              >
                当前会话
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={showingSnapshot}
                onClick={() =>
                  setSelectedSnapshotId(snapshot.snapshot_id)
                }
                className={`rounded px-3 py-1.5 text-sm ${
                  showingSnapshot
                    ? "bg-surface font-semibold text-text shadow-sm"
                    : "text-text-muted hover:text-text"
                }`}
              >
                本轮输入
              </button>
            </div>
          )}

          {!showingSnapshot && !browsingConversation && (
            <button
              type="button"
              disabled={!task.trim() || previewLoading}
              onClick={() => void previewContext(task)}
              className="mb-4 inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm font-medium hover:bg-surface-alt disabled:opacity-50"
            >
              {previewLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              刷新预览
            </button>
          )}

          {error && (
            <div
              role="alert"
              className="mb-4 border-l-2 border-red-500 pl-3 text-sm text-red-700 dark:text-red-300"
            >
              {error}
            </div>
          )}

          {previewLoading && !displayed && !browsingConversation && (
            <div className="flex items-center gap-2 py-10 text-sm text-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              正在构建上下文…
            </div>
          )}

          {browsingConversation && conversationItems.length === 0 && (
            <div className="py-10 text-sm text-text-muted">
              当前会话还没有可查看的历史消息。
            </div>
          )}

          {browsingConversation && conversationItems.length > 0 && (
            <>
              <div className="mb-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border">
                <div className="bg-surface px-3 py-2">
                  <p className="text-[11px] text-text-muted">会话消息</p>
                  <p className="text-sm font-semibold">
                    {conversationItems.length} 条
                  </p>
                </div>
                <div className="bg-surface px-3 py-2">
                  <p className="text-[11px] text-text-muted">可用消息</p>
                  <p className="text-sm font-semibold">
                    {
                      conversationItems.filter(
                        (item) => item.metadata.context_eligible === true,
                      ).length
                    }{" "}
                    条
                  </p>
                </div>
              </div>
              <section>
                <h3 className="text-xs font-semibold text-text-muted">
                  当前会话 · {conversationItems.length}
                </h3>
                <ul className="mt-1">
                  {conversationItems.map((item) => {
                    const contextEligible =
                      item.metadata.context_eligible === true;
                    const selectionDisabledReason = !contextEligible
                      ? "这条消息已被遗忘，不能重新加入上下文；仍可查看或彻底删除。"
                      : independent
                        ? "独立研究已开启，下一次研究不会读取历史上下文。"
                        : contextMode === "disabled"
                          ? "上下文模式已关闭，下一次研究不会读取历史上下文。"
                          : undefined;
                    return (
                      <ContextRow
                        key={item.context_id}
                        item={item}
                        browseMode
                        selectionDisabledReason={selectionDisabledReason}
                        onToggle={(included) =>
                          toggleContextItem(item.context_id, included)
                        }
                        onPin={(pinned) =>
                          setPinned(
                            item.source_type,
                            item.source_id,
                            pinned,
                          )
                        }
                        onForget={() => {
                          if (
                            window.confirm(
                              "将保留本轮问题与回答供查看，但以后不再用于上下文或记忆。继续吗？",
                            )
                          ) {
                            return forgetMessage(item.source_id);
                          }
                          return undefined;
                        }}
                        onDelete={() => {
                          if (
                            window.confirm(
                              "将彻底删除本轮问题、回答及其衍生摘要、产物和记忆。此操作不可撤销。",
                            )
                          ) {
                            return deleteMessage(item.source_id);
                          }
                          return undefined;
                        }}
                      />
                    );
                  })}
                </ul>
              </section>
            </>
          )}

          {displayed && (
            <>
              <div className="mb-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border">
                <div className="bg-surface px-3 py-2">
                  <p className="text-[11px] text-text-muted">预算</p>
                  <p className="text-sm font-semibold">
                    {displayed.used_tokens}/{displayed.budget_tokens} tokens
                  </p>
                </div>
                <div className="bg-surface px-3 py-2">
                  <p className="text-[11px] text-text-muted">已使用</p>
                  <p className="text-sm font-semibold">
                    {displayed.items.length} 项
                  </p>
                </div>
              </div>

              {displayed.items.length === 0 ? (
                <div className="flex items-center gap-2 py-8 text-sm text-text-muted">
                  <Check className="h-4 w-4" />
                  本次研究不使用历史上下文。
                </div>
              ) : (
                <div className="space-y-5">
                  {grouped.map(([sourceType, items]) => (
                    <section key={sourceType}>
                      <h3 className="text-xs font-semibold text-text-muted">
                        {SOURCE_LABELS[sourceType]} · {items.length}
                      </h3>
                      <ul className="mt-1">
                        {items.map((item) => (
                          <ContextRow
                            key={item.context_id}
                            item={item}
                            selectionDisabledReason={
                              showingSnapshot
                                ? "这是本次研究实际使用的上下文快照，运行完成后不能再修改。"
                                : undefined
                            }
                            actionsDisabledReason={
                              showingSnapshot
                                ? "本轮输入快照不可修改；切换到“当前会话”可固定、遗忘或删除原消息。"
                                : undefined
                            }
                            onToggle={(included) => {
                              toggleContextItem(item.context_id, included);
                              void previewContext(task);
                            }}
                            onPin={(pinned) =>
                              setPinned(
                                item.source_type,
                                item.source_id,
                                pinned,
                              )
                            }
                            onForget={() => {
                              if (
                                window.confirm(
                                  "将保留本轮问题与回答供查看，但以后不再用于上下文或记忆。继续吗？",
                                )
                              ) {
                                return forgetMessage(item.source_id).then(() =>
                                  previewContext(task),
                                );
                              }
                              return undefined;
                            }}
                            onDelete={() => {
                              if (
                                window.confirm(
                                  "将彻底删除本轮问题、回答及其衍生摘要、产物和记忆。此操作不可撤销。",
                                )
                              ) {
                                return deleteMessage(item.source_id).then(() =>
                                  previewContext(task),
                                );
                              }
                              return undefined;
                            }}
                          />
                        ))}
                      </ul>
                    </section>
                  ))}
                </div>
              )}

              {hasTask && preview && preview.excluded.length > 0 && (
                <details className="mt-5 border-t border-border pt-4">
                  <summary className="cursor-pointer text-xs font-semibold text-text-muted">
                    未使用项 · {preview.excluded.length}
                  </summary>
                  <ul className="mt-2">
                    {preview.excluded.map((item) => (
                      <ContextRow
                        key={item.context_id}
                        item={item}
                        onToggle={(included) => {
                          toggleContextItem(item.context_id, included);
                          void previewContext(task);
                        }}
                        onPin={(pinned) =>
                          setPinned(
                            item.source_type,
                            item.source_id,
                            pinned,
                          )
                        }
                        onForget={() => forgetMessage(item.source_id)}
                        onDelete={() => deleteMessage(item.source_id)}
                      />
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
