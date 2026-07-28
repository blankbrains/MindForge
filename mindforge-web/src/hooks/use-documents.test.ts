import { afterEach, describe, expect, it, vi } from "vitest";

import {
  friendlyError,
  uploadIndexJob,
} from "@/hooks/use-documents";

class MockXMLHttpRequest {
  static instance: MockXMLHttpRequest;

  status = 202;
  responseText = "";
  upload = {
    onprogress: null as ((event: ProgressEvent) => void) | null,
  };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  open = vi.fn();
  send = vi.fn();
  abort = vi.fn(() => this.onabort?.());

  constructor() {
    MockXMLHttpRequest.instance = this;
  }
}

afterEach(() => vi.unstubAllGlobals());

describe("friendlyError", () => {
  it("preserves validated parser details instead of masking them", () => {
    expect(
      friendlyError(
        413,
        JSON.stringify({
          detail: "PDF 共 523 页，超过当前上限 500 页。",
        }),
      ),
    ).toBe("PDF 共 523 页，超过当前上限 500 页。");
  });

  it("reports byte-level upload progress and returns the created job", async () => {
    vi.stubGlobal(
      "XMLHttpRequest",
      MockXMLHttpRequest as unknown as typeof XMLHttpRequest,
    );
    const progress = vi.fn();
    const formData = new FormData();
    formData.append(
      "file",
      new File(["0123456789"], "document.pdf", {
        type: "application/pdf",
      }),
    );

    const resultPromise = uploadIndexJob({
      formData,
      onUploadProgress: progress,
    });
    const request = MockXMLHttpRequest.instance;
    request.upload.onprogress?.({
      lengthComputable: true,
      loaded: 5,
      total: 10,
    } as ProgressEvent);
    request.responseText = JSON.stringify({ job_id: "job-1" });
    request.onload?.();

    await expect(resultPromise).resolves.toMatchObject({ job_id: "job-1" });
    expect(progress).toHaveBeenCalledWith({
      loaded: 5,
      total: 10,
      percent: 50,
    });
    expect(progress).toHaveBeenLastCalledWith({
      loaded: 10,
      total: 10,
      percent: 100,
    });
  });

  it("aborts the active upload through AbortSignal", async () => {
    vi.stubGlobal(
      "XMLHttpRequest",
      MockXMLHttpRequest as unknown as typeof XMLHttpRequest,
    );
    const controller = new AbortController();
    const resultPromise = uploadIndexJob({
      formData: new FormData(),
      signal: controller.signal,
    });

    controller.abort();

    await expect(resultPromise).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(MockXMLHttpRequest.instance.abort).toHaveBeenCalledOnce();
  });
});
