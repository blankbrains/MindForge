import { create } from "zustand";
import { api } from "@/lib/api";
import type {
  ContextMode,
  ContextPreview,
  ContextSnapshot,
  Conversation,
  ConversationDetail,
} from "@/types/context";

let loadGeneration = 0;
let refreshGeneration = 0;
let previewGeneration = 0;
let snapshotGeneration = 0;

interface ContextState {
  conversations: Conversation[];
  activeConversationId: string | null;
  activeConversation: ConversationDetail | null;
  contextMode: ContextMode;
  independent: boolean;
  selectedContextIds: string[];
  excludedContextIds: string[];
  drawerOpen: boolean;
  loading: boolean;
  previewLoading: boolean;
  error: string | null;
  preview: ContextPreview | null;
  snapshot: ContextSnapshot | null;

  initialize: () => Promise<void>;
  createConversation: () => Promise<void>;
  selectConversation: (conversationId: string) => Promise<void>;
  refreshConversation: () => Promise<void>;
  setContextMode: (mode: ContextMode) => Promise<void>;
  setIndependent: (independent: boolean) => void;
  setDrawerOpen: (open: boolean) => void;
  toggleContextItem: (contextId: string, included: boolean) => void;
  previewContext: (task: string) => Promise<void>;
  setPinned: (
    sourceType: string,
    sourceId: string,
    pinned: boolean,
  ) => Promise<void>;
  forgetMessage: (messageId: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  loadSnapshot: (runId: string) => Promise<void>;
  deleteActiveConversation: () => Promise<void>;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "上下文请求失败。";
}

function errorStatus(error: unknown): number | null {
  if (
    typeof error === "object"
    && error !== null
    && "status" in error
    && typeof (error as { status?: unknown }).status === "number"
  ) {
    return (error as { status: number }).status;
  }
  return null;
}

function latestRunId(conversation: ConversationDetail): string | null {
  for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
    const runId = conversation.messages[index]?.run_id;
    if (typeof runId === "string" && runId.trim()) {
      return runId;
    }
  }
  return null;
}

function affectedMessageContextIds(
  conversation: ConversationDetail | null,
  messageId: string,
): Set<string> {
  const target = conversation?.messages.find(
    (message) => message.message_id === messageId,
  );
  const affectedIds =
    target?.run_id
      ? conversation?.messages
          .filter((message) => message.run_id === target.run_id)
          .map((message) => message.message_id)
      : [messageId];
  return new Set(
    (affectedIds ?? [messageId]).flatMap((id) => [id, `message:${id}`]),
  );
}

async function restoreLatestSnapshot(
  conversation: ConversationDetail,
): Promise<{ snapshot: ContextSnapshot | null; error: string | null }> {
  const runId = latestRunId(conversation);
  if (!runId) return { snapshot: null, error: null };
  try {
    const snapshot = await api.get<ContextSnapshot>(
      `/research-runs/${runId}/context`,
    );
    return { snapshot, error: null };
  } catch (error) {
    if (errorStatus(error) === 404) {
      return { snapshot: null, error: null };
    }
    return {
      snapshot: null,
      error: `最近一次上下文快照恢复失败：${errorMessage(error)}`,
    };
  }
}

export const useContextStore = create<ContextState>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  activeConversation: null,
  contextMode: "auto",
  independent: false,
  selectedContextIds: [],
  excludedContextIds: [],
  drawerOpen: false,
  loading: false,
  previewLoading: false,
  error: null,
  preview: null,
  snapshot: null,

  initialize: async () => {
    const generation = ++loadGeneration;
    const snapshotRequestGeneration = ++snapshotGeneration;
    set({ loading: true, error: null });
    try {
      let conversations = await api.get<Conversation[]>("/conversations");
      if (generation !== loadGeneration) return;
      if (conversations.length === 0) {
        const created = await api.post<Conversation>("/conversations", {
          title: "新研究",
          context_mode: "auto",
        });
        conversations = [created];
      }
      const currentId = get().activeConversationId;
      const activeId =
        conversations.find((item) => item.conversation_id === currentId)
          ?.conversation_id ?? conversations[0].conversation_id;
      const detail = await api.get<ConversationDetail>(
        `/conversations/${activeId}`,
      );
      const restored = await restoreLatestSnapshot(detail);
      if (generation !== loadGeneration) return;
      set({
        conversations,
        activeConversationId: activeId,
        activeConversation: detail,
        contextMode: detail.context_mode,
        snapshot:
          snapshotRequestGeneration === snapshotGeneration
            ? restored.snapshot
            : get().snapshot,
        loading: false,
        error:
          snapshotRequestGeneration === snapshotGeneration
            ? restored.error
            : get().error,
      });
    } catch (error) {
      if (generation !== loadGeneration) return;
      set({ loading: false, error: errorMessage(error) });
    }
  },

  createConversation: async () => {
    const generation = ++loadGeneration;
    previewGeneration += 1;
    snapshotGeneration += 1;
    set({ loading: true, error: null });
    try {
      const created = await api.post<Conversation>("/conversations", {
        title: "新研究",
        context_mode: "auto",
      });
      const detail = await api.get<ConversationDetail>(
        `/conversations/${created.conversation_id}`,
      );
      if (generation !== loadGeneration) return;
      set((state) => ({
        conversations: [created, ...state.conversations],
        activeConversationId: created.conversation_id,
        activeConversation: detail,
        contextMode: created.context_mode,
        independent: false,
        selectedContextIds: [],
        excludedContextIds: [],
        preview: null,
        snapshot: null,
        loading: false,
      }));
    } catch (error) {
      if (generation !== loadGeneration) return;
      set({ loading: false, error: errorMessage(error) });
    }
  },

  selectConversation: async (conversationId) => {
    if (conversationId === get().activeConversationId) return;
    const generation = ++loadGeneration;
    previewGeneration += 1;
    const snapshotRequestGeneration = ++snapshotGeneration;
    set({
      loading: true,
      error: null,
      preview: null,
      snapshot: null,
    });
    try {
      const detail = await api.get<ConversationDetail>(
        `/conversations/${conversationId}`,
      );
      const restored = await restoreLatestSnapshot(detail);
      if (generation !== loadGeneration) return;
      set({
        activeConversationId: conversationId,
        activeConversation: detail,
        contextMode: detail.context_mode,
        independent: false,
        selectedContextIds: [],
        excludedContextIds: [],
        preview: null,
        snapshot:
          snapshotRequestGeneration === snapshotGeneration
            ? restored.snapshot
            : get().snapshot,
        loading: false,
        error:
          snapshotRequestGeneration === snapshotGeneration
            ? restored.error
            : get().error,
      });
    } catch (error) {
      if (generation !== loadGeneration) return;
      set({ loading: false, error: errorMessage(error) });
    }
  },

  refreshConversation: async () => {
    const conversationId = get().activeConversationId;
    if (!conversationId) return;
    const generation = ++refreshGeneration;
    try {
      const [detail, conversations] = await Promise.all([
        api.get<ConversationDetail>(`/conversations/${conversationId}`),
        api.get<Conversation[]>("/conversations"),
      ]);
      if (
        generation !== refreshGeneration
        || get().activeConversationId !== conversationId
      ) {
        return;
      }
      set({
        activeConversation: detail,
        conversations,
        contextMode: detail.context_mode,
        error: null,
      });
    } catch (error) {
      if (
        generation !== refreshGeneration
        || get().activeConversationId !== conversationId
      ) {
        return;
      }
      set({ error: errorMessage(error) });
    }
  },

  setContextMode: async (mode) => {
    const conversation = get().activeConversation;
    if (!conversation) return;
    const previous = get().contextMode;
    previewGeneration += 1;
    snapshotGeneration += 1;
    set({
      contextMode: mode,
      selectedContextIds: mode === "manual" ? get().selectedContextIds : [],
      excludedContextIds: mode === "disabled" ? [] : get().excludedContextIds,
      preview: null,
      snapshot: null,
    });
    try {
      const updated = await api.patch<Conversation>(
        `/conversations/${conversation.conversation_id}`,
        {
          context_mode: mode,
          version: conversation.version,
        },
      );
      set((state) => ({
        activeConversation: state.activeConversation
          ? { ...state.activeConversation, ...updated }
          : null,
        conversations: state.conversations.map((item) =>
          item.conversation_id === updated.conversation_id ? updated : item
        ),
      }));
    } catch (error) {
      set({ contextMode: previous, error: errorMessage(error) });
    }
  },

  setIndependent: (independent) =>
    {
      previewGeneration += 1;
      snapshotGeneration += 1;
      set({
        independent,
        preview: null,
        snapshot: null,
      });
    },

  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),

  toggleContextItem: (contextId, included) =>
    {
      previewGeneration += 1;
      set((state) => {
      if (state.contextMode === "manual") {
        return {
          selectedContextIds: included
            ? Array.from(new Set([...state.selectedContextIds, contextId]))
            : state.selectedContextIds.filter((item) => item !== contextId),
          excludedContextIds: state.excludedContextIds.filter(
            (item) => item !== contextId,
          ),
        };
      }
      return {
        excludedContextIds: included
          ? state.excludedContextIds.filter((item) => item !== contextId)
          : Array.from(new Set([...state.excludedContextIds, contextId])),
      };
      });
    },

  previewContext: async (task) => {
    const generation = ++previewGeneration;
    snapshotGeneration += 1;
    const state = get();
    if (!state.activeConversationId || !task.trim()) {
      set({ preview: null, previewLoading: false });
      return;
    }
    const conversationId = state.activeConversationId;
    set({
      previewLoading: true,
      error: null,
      preview: null,
      snapshot: null,
    });
    try {
      const preview = await api.post<ContextPreview>(
        `/conversations/${conversationId}/context-preview`,
        {
          task: task.trim(),
          context_mode: state.contextMode,
          selected_context_ids: state.selectedContextIds,
          excluded_context_ids: state.excludedContextIds,
          independent: state.independent,
        },
      );
      if (
        generation !== previewGeneration
        || get().activeConversationId !== conversationId
      ) {
        return;
      }
      set({ preview, previewLoading: false });
    } catch (error) {
      if (
        generation !== previewGeneration
        || get().activeConversationId !== conversationId
      ) {
        return;
      }
      set({ previewLoading: false, error: errorMessage(error) });
    }
  },

  setPinned: async (sourceType, sourceId, pinned) => {
    const state = get();
    if (!state.activeConversationId) return;
    try {
      const query =
        sourceType === "message"
          ? `?conversation_id=${state.activeConversationId}`
          : "";
      await api.patch(`/context-items/${sourceType}/${sourceId}${query}`, {
        pinned,
      });
      await get().refreshConversation();
      if (state.preview) {
        set((current) => ({
          preview: current.preview
            ? {
                ...current.preview,
                items: current.preview.items.map((item) =>
                  item.source_type === sourceType
                  && item.source_id === sourceId
                    ? { ...item, pinned }
                    : item
                ),
              }
            : null,
        }));
      }
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  forgetMessage: async (messageId) => {
    const current = get();
    const conversationId = current.activeConversationId;
    if (!conversationId) return;
    const affectedIds = affectedMessageContextIds(
      current.activeConversation,
      messageId,
    );
    try {
      await api.post(
        `/conversations/${conversationId}/messages/${messageId}/forget`,
      );
      previewGeneration += 1;
      set((state) => ({
        selectedContextIds: state.selectedContextIds.filter(
          (item) => !affectedIds.has(item),
        ),
        excludedContextIds: state.excludedContextIds.filter(
          (item) => !affectedIds.has(item),
        ),
        preview: null,
      }));
      await get().refreshConversation();
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  deleteMessage: async (messageId) => {
    const current = get();
    const conversationId = current.activeConversationId;
    if (!conversationId) return;
    const target = current.activeConversation?.messages.find(
      (message) => message.message_id === messageId,
    );
    const affectedIds = affectedMessageContextIds(
      current.activeConversation,
      messageId,
    );
    try {
      await api.delete(
        `/conversations/${conversationId}/messages/${messageId}`,
      );
      previewGeneration += 1;
      snapshotGeneration += 1;
      set((state) => ({
        selectedContextIds: state.selectedContextIds.filter(
          (item) => !affectedIds.has(item),
        ),
        excludedContextIds: state.excludedContextIds.filter(
          (item) => !affectedIds.has(item),
        ),
        preview: null,
        snapshot:
          target?.run_id && state.snapshot?.run_id === target.run_id
            ? null
            : state.snapshot,
      }));
      await get().refreshConversation();
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  loadSnapshot: async (runId) => {
    const generation = ++snapshotGeneration;
    set({ previewLoading: true, error: null });
    try {
      const snapshot = await api.get<ContextSnapshot>(
        `/research-runs/${runId}/context`,
      );
      if (generation !== snapshotGeneration) return;
      set({ snapshot, previewLoading: false });
    } catch (error) {
      if (generation !== snapshotGeneration) return;
      set({ previewLoading: false, error: errorMessage(error) });
    }
  },

  deleteActiveConversation: async () => {
    const conversationId = get().activeConversationId;
    if (!conversationId) return;
    const generation = ++loadGeneration;
    previewGeneration += 1;
    snapshotGeneration += 1;
    set({ loading: true, error: null });
    try {
      await api.delete(`/conversations/${conversationId}`);
      if (generation !== loadGeneration) return;
      set({
        activeConversationId: null,
        activeConversation: null,
        conversations: [],
        preview: null,
        snapshot: null,
      });
      await get().initialize();
    } catch (error) {
      if (generation !== loadGeneration) return;
      set({ loading: false, error: errorMessage(error) });
    }
  },
}));
