import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ContextDrawer } from "@/components/research/context-drawer";
import { useContextStore } from "@/store/context-store";

const originalPreviewContext = useContextStore.getState().previewContext;

afterEach(() => {
  cleanup();
  useContextStore.setState({ previewContext: originalPreviewContext });
});

describe("ContextDrawer", () => {
  beforeEach(() => {
    useContextStore.setState({
      preview: null,
      snapshot: null,
      previewLoading: false,
      error: null,
      contextMode: "auto",
      independent: false,
      previewContext: vi.fn().mockResolvedValue(undefined),
    });
  });

  it("does not refresh again when the parent rerenders with stable controls", () => {
    const previewContext = useContextStore.getState().previewContext;
    const onClose = vi.fn();
    const { rerender } = render(
      <ContextDrawer open task="follow-up" onClose={onClose} />,
    );

    rerender(<ContextDrawer open task="follow-up" onClose={onClose} />);

    expect(previewContext).toHaveBeenCalledOnce();
  });
});
