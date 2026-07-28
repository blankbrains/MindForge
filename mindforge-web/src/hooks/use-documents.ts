import { useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { API_BASE } from "@/lib/constants";
import type { DocumentItem, IndexJob } from "@/types/document";

export interface UploadProgress {
  loaded: number;
  total: number | null;
  percent: number | null;
}

interface UploadIndexJobOptions {
  formData: FormData;
  signal?: AbortSignal;
  onUploadProgress?: (progress: UploadProgress) => void;
}

/** 将后端原始错误转为用户友好的中文提示 */
export function friendlyError(status: number, raw: string): string {
  // Prefer the server's validated detail, including the configured size limit.
  try {
    const parsed = JSON.parse(raw);
    if (parsed.detail) return String(parsed.detail);
  } catch {
    // Non-JSON error responses are handled by the raw-text fallback below.
  }
  if (status >= 500) return "服务器繁忙，请稍后重试";
  if (status === 413) return "文件过大，请压缩后重试";
  if (status === 400) return "文件格式不支持，请上传 PDF、DOCX、TXT、HTML 或 Markdown 文件";
  if (status === 422) return "上传参数有误，请刷新页面后重试";
  if (raw && raw.length < 100) return raw;
  return "上传失败，请检查网络连接后重试";
}

function createAbortError(): Error {
  const error = new Error("上传已取消");
  error.name = "AbortError";
  return error;
}

export function uploadIndexJob({
  formData,
  signal,
  onUploadProgress,
}: UploadIndexJobOptions): Promise<IndexJob> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortError());
      return;
    }

    const request = new XMLHttpRequest();
    const uploadedFile = formData.get("file");
    const uploadedSize =
      uploadedFile instanceof File ? uploadedFile.size : null;
    let settled = false;

    const cleanup = () => {
      signal?.removeEventListener("abort", handleSignalAbort);
    };
    const resolveOnce = (job: IndexJob) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(job);
    };
    const rejectOnce = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const handleSignalAbort = () => request.abort();

    request.open("POST", `${API_BASE}/index-jobs`);
    request.upload.onprogress = (event) => {
      const total =
        event.lengthComputable && event.total > 0 ? event.total : null;
      onUploadProgress?.({
        loaded: event.loaded,
        total,
        percent:
          total === null
            ? null
            : Math.min(100, Math.max(0, (event.loaded / total) * 100)),
      });
    };
    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        rejectOnce(
          new Error(friendlyError(request.status, request.responseText)),
        );
        return;
      }
      try {
        const job = JSON.parse(request.responseText) as IndexJob;
        onUploadProgress?.({
          loaded: uploadedSize ?? 0,
          total: uploadedSize,
          percent: 100,
        });
        resolveOnce(job);
      } catch {
        rejectOnce(new Error("服务器返回了无法识别的上传结果"));
      }
    };
    request.onerror = () => {
      rejectOnce(new Error("上传失败，请检查网络连接后重试"));
    };
    request.onabort = () => {
      rejectOnce(createAbortError());
    };

    signal?.addEventListener("abort", handleSignalAbort, { once: true });
    try {
      request.send(formData);
    } catch {
      rejectOnce(new Error("无法启动文件上传，请刷新页面后重试"));
    }
  });
}

export function useDocuments(indexJobId: string | null = null) {
  const qc = useQueryClient();
  const observedTerminalJob = useRef<string | null>(null);

  const list = useQuery<DocumentItem[]>({
    queryKey: ["documents"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) throw new Error("文档列表加载失败，请刷新页面重试");
      return res.json();
    },
  });

  const job = useQuery<IndexJob>({
    queryKey: ["index-job", indexJobId],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/index-jobs/${indexJobId}`);
      if (!res.ok) {
        throw new Error("索引进度加载失败，请稍后重试");
      }
      return res.json();
    },
    enabled: Boolean(indexJobId),
    refetchInterval: (query) =>
      query.state.data?.status === "queued"
      || query.state.data?.status === "running"
        ? 1_000
        : false,
  });

  useEffect(() => {
    if (!job.data) return;
    const identity = `${job.data.job_id}:${job.data.status}:${job.data.updated_at}`;
    if (
      !["completed", "failed", "cancelled"].includes(job.data.status)
      || observedTerminalJob.current === identity
    ) return;
    observedTerminalJob.current = identity;
    void qc.invalidateQueries({ queryKey: ["stats"] });
    void qc.invalidateQueries({ queryKey: ["documents"] });
  }, [job.data, qc]);

  const upload = useMutation({
    mutationFn: uploadIndexJob,
  });

  const cancelJob = useMutation({
    mutationFn: async (jobId: string) => {
      const res = await fetch(`${API_BASE}/index-jobs/${jobId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(friendlyError(res.status, text));
      }
      return res.json() as Promise<IndexJob>;
    },
    onSuccess: (cancelledJob) => {
      qc.setQueryData(
        ["index-job", cancelledJob.job_id],
        cancelledJob,
      );
    },
  });

  const remove = useMutation({
    mutationFn: async (docId: string) => {
      const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(friendlyError(res.status, text));
      }
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  return { list, job, upload, cancelJob, remove };
}
