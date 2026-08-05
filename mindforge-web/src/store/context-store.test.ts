import { beforeEach, describe, expect, it, vi } from "vitest";

const apiState = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  patch: vi.fn(),
  delete: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiState,
}));

import { useContextStore } from "@/store/context-store";

const conversation = {
  conversation_id: "a".repeat(32),
  title: "Context design",
  status: "active" as const,
  context_mode: "auto" as const,
  version: 1,
  created_at: "2026-08-05T00:00:00Z",
  updated_at: "2026-08-05T00:00:00Z",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

describe("context store", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useContextStore.setState({
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
    });
  });

  it("initializes the active conversation from the server", async () => {
    apiState.get
      .mockResolvedValueOnce([conversation])
      .mockResolvedValueOnce({ ...conversation, messages: [] });

    await useContextStore.getState().initialize();

    expect(useContextStore.getState().activeConversationId).toBe(
      conversation.conversation_id,
    );
    expect(useContextStore.getState().contextMode).toBe("auto");
  });

  it("keeps manual inclusion and one-run exclusion mutually exclusive", () => {
    useContextStore.setState({ contextMode: "manual" });

    useContextStore
      .getState()
      .toggleContextItem("message:m1", true);

    expect(useContextStore.getState().selectedContextIds).toEqual([
      "message:m1",
    ]);
    expect(useContextStore.getState().excludedContextIds).toEqual([]);

    useContextStore
      .getState()
      .toggleContextItem("message:m1", false);
    expect(useContextStore.getState().selectedContextIds).toEqual([]);
  });

  it("sends the exact preview controls", async () => {
    useContextStore.setState({
      activeConversationId: conversation.conversation_id,
      contextMode: "manual",
      selectedContextIds: ["message:m1"],
      excludedContextIds: ["artifact:a1"],
    });
    apiState.post.mockResolvedValue({
      snapshot_id: null,
      standalone_query: "resolved",
      requires_context: true,
      budget_tokens: 4000,
      used_tokens: 10,
      context_fingerprint: "f",
      policy_version: "v1",
      embedding_version: "lexical-v1",
      items: [],
      excluded: [],
    });

    await useContextStore.getState().previewContext("follow-up");

    expect(apiState.post).toHaveBeenCalledWith(
      `/conversations/${conversation.conversation_id}/context-preview`,
      {
        task: "follow-up",
        context_mode: "manual",
        selected_context_ids: ["message:m1"],
        excluded_context_ids: ["artifact:a1"],
        independent: false,
      },
    );
  });

  it("does not let a stale conversation response replace the latest selection", async () => {
    const second = deferred<typeof conversation & { messages: [] }>();
    const third = deferred<typeof conversation & { messages: [] }>();
    const secondId = "b".repeat(32);
    const thirdId = "c".repeat(32);
    useContextStore.setState({
      conversations: [
        conversation,
        { ...conversation, conversation_id: secondId, title: "Second" },
        { ...conversation, conversation_id: thirdId, title: "Third" },
      ],
      activeConversationId: conversation.conversation_id,
      activeConversation: { ...conversation, messages: [] },
    });
    apiState.get
      .mockReturnValueOnce(second.promise)
      .mockReturnValueOnce(third.promise);

    const secondRequest = useContextStore.getState().selectConversation(secondId);
    const thirdRequest = useContextStore.getState().selectConversation(thirdId);
    third.resolve({
      ...conversation,
      conversation_id: thirdId,
      title: "Third",
      messages: [],
    });
    await thirdRequest;
    second.resolve({
      ...conversation,
      conversation_id: secondId,
      title: "Second",
      messages: [],
    });
    await secondRequest;

    expect(useContextStore.getState().activeConversationId).toBe(thirdId);
    expect(useContextStore.getState().activeConversation?.title).toBe("Third");
  });

  it("does not let a stale preview replace the latest task preview", async () => {
    const first = deferred<{
      snapshot_id: null;
      standalone_query: string;
      requires_context: boolean;
      budget_tokens: number;
      used_tokens: number;
      context_fingerprint: string;
      policy_version: string;
      embedding_version: string;
      items: [];
      excluded: [];
    }>();
    const second = deferred<{
      snapshot_id: null;
      standalone_query: string;
      requires_context: boolean;
      budget_tokens: number;
      used_tokens: number;
      context_fingerprint: string;
      policy_version: string;
      embedding_version: string;
      items: [];
      excluded: [];
    }>();
    useContextStore.setState({
      activeConversationId: conversation.conversation_id,
    });
    apiState.post
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const firstRequest = useContextStore.getState().previewContext("first");
    const secondRequest = useContextStore.getState().previewContext("second");
    second.resolve({
      snapshot_id: null,
      standalone_query: "second",
      requires_context: false,
      budget_tokens: 4000,
      used_tokens: 0,
      context_fingerprint: "second",
      policy_version: "v1",
      embedding_version: "lexical-v1",
      items: [],
      excluded: [],
    });
    await secondRequest;
    first.resolve({
      snapshot_id: null,
      standalone_query: "first",
      requires_context: false,
      budget_tokens: 4000,
      used_tokens: 0,
      context_fingerprint: "first",
      policy_version: "v1",
      embedding_version: "lexical-v1",
      items: [],
      excluded: [],
    });
    await firstRequest;

    expect(useContextStore.getState().preview?.standalone_query).toBe("second");
  });

  it("clears an obsolete preview while a replacement is loading", async () => {
    const next = deferred<never>();
    useContextStore.setState({
      activeConversationId: conversation.conversation_id,
      preview: {
        snapshot_id: null,
        standalone_query: "obsolete",
        requires_context: false,
        budget_tokens: 4000,
        used_tokens: 0,
        context_fingerprint: "old",
        policy_version: "v1",
        embedding_version: "lexical-v1",
        items: [],
        excluded: [],
      },
    });
    apiState.post.mockReturnValueOnce(next.promise);

    void useContextStore.getState().previewContext("replacement");

    expect(useContextStore.getState().preview).toBeNull();
    expect(useContextStore.getState().previewLoading).toBe(true);
  });
});
