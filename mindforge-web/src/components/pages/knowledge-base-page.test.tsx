import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const documentState = vi.hoisted(() => ({
  documents: [
    { doc_id: "first", filename: "first.txt", chunk_count: 1, status: "indexed" },
    { doc_id: "second", filename: "second.txt", chunk_count: 1, status: "indexed" },
  ],
}));
const runningJob = vi.hoisted(() => ({
  enabled: false,
}));
const uploadState = vi.hoisted(() => ({
  pending: false,
  completeImmediately: true,
}));

vi.mock("@/hooks/use-stats", () => ({
  useStats: () => ({
    data: {
      documents_indexed: 2,
      chunks_indexed: 2,
      qdrant_connected: true,
      qdrant_url: "",
      redis_url: "",
      max_upload_mb: 200,
      max_pdf_pages: 600,
    },
    isLoading: false,
  }),
}));

vi.mock("@/hooks/use-documents", () => ({
  useDocuments: (jobId: string | null) => ({
    list: {
      data: documentState.documents,
      isLoading: false,
      isError: false,
    },
    job: {
      data: jobId && runningJob.enabled
        ? {
            job_id: jobId,
            doc_id: null,
            filename: "large.pdf",
            status: "running",
            stage: "embedding",
            progress: 25,
            chunk_count: 925,
            timings: {},
            error: null,
            cancel_requested: false,
            strategy: "auto",
            use_raptor: false,
            use_graphrag: false,
            created_at: "2026-07-28T00:00:00Z",
            updated_at: "2026-07-28T00:00:01Z",
          }
        : undefined,
      isError: false,
    },
    upload: {
      isPending: uploadState.pending,
      mutate: vi.fn((variables, options) => {
        uploadState.pending = true;
        variables.onUploadProgress?.({
          loaded: 42,
          total: 100,
          percent: 42,
        });
        if (uploadState.completeImmediately) {
          uploadState.pending = false;
          runningJob.enabled = true;
          options?.onSuccess?.({ job_id: "job-1" });
        }
      }),
    },
    cancelJob: {
      isPending: false,
      mutate: vi.fn(),
    },
    remove: {
      isPending: false,
      mutateAsync: vi.fn(),
    },
  }),
}));

import { KnowledgeBasePage } from "./knowledge-base-page";

afterEach(() => cleanup());

describe("KnowledgeBasePage", () => {
  it("shows only the current upload progress, not an index-job history", async () => {
    runningJob.enabled = false;
    uploadState.pending = false;
    uploadState.completeImmediately = true;
    render(<KnowledgeBasePage />);

    expect(screen.queryByText("索引任务")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "上传文档" }));
    const fileInput = screen.getByLabelText("选择文件");
    fireEvent.change(fileInput, {
      target: {
        files: [new File(["content"], "large.pdf", { type: "application/pdf" })],
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始索引" }));

    expect(await screen.findByText("正在生成向量")).not.toBeNull();
    expect(screen.getByText("25% · 925 块")).not.toBeNull();
    expect(screen.queryByText("索引任务")).toBeNull();
  });

  it("shows real file-transfer progress before indexing starts", async () => {
    runningJob.enabled = false;
    uploadState.pending = false;
    uploadState.completeImmediately = false;
    render(<KnowledgeBasePage />);

    fireEvent.click(screen.getByRole("button", { name: "上传文档" }));
    fireEvent.change(screen.getByLabelText("选择文件"), {
      target: {
        files: [new File(["content"], "large.pdf", {
          type: "application/pdf",
        })],
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "开始索引" }));

    expect(await screen.findByText("正在上传到服务器")).not.toBeNull();
    expect(screen.getByText("42%")).not.toBeNull();
    expect(
      screen
        .getByRole("progressbar", { name: "文档上传进度" })
        .getAttribute("aria-valuenow"),
    ).toBe("42");
    expect(
      screen.getByRole("button", { name: "取消上传" }),
    ).not.toBeNull();
  });

  it("ignores a stale document preview response", async () => {
    let resolveFirst!: (response: Response) => void;
    let resolveSecond!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      return new Promise<Response>((resolve) => {
        if (url.includes("/first/")) {
          resolveFirst = resolve;
        } else {
          resolveSecond = resolve;
        }
      });
    });

    render(<KnowledgeBasePage />);
    fireEvent.click(screen.getByText("first.txt"));
    fireEvent.click(screen.getByText("second.txt"));

    resolveSecond(
      new Response(JSON.stringify({ content: "second content" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    expect(await screen.findByText("second content")).not.toBeNull();

    resolveFirst(
      new Response(JSON.stringify({ content: "stale first content" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await waitFor(() => {
      expect(screen.queryByText("stale first content")).toBeNull();
      expect(screen.getByText("second content")).not.toBeNull();
    });
  });
});
