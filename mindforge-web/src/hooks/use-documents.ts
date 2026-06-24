import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { API_BASE } from "@/lib/constants";
import type { DocumentItem } from "@/types/document";

/** 将后端原始错误转为用户友好的中文提示 */
function friendlyError(status: number, raw: string): string {
  if (status === 413) return "文件过大，请压缩后重试（最大支持 200MB）";
  if (status === 400) return "文件格式不支持，请上传 PDF、DOCX、TXT 或 Markdown 文件";
  if (status === 422) return "上传参数有误，请刷新页面后重试";
  if (status >= 500) return "服务器繁忙，请稍后重试";
  // 尝试从原始响应中提取有意义的信息
  try {
    const parsed = JSON.parse(raw);
    if (parsed.detail) return String(parsed.detail);
  } catch {}
  if (raw && raw.length < 100) return raw;
  return "上传失败，请检查网络连接后重试";
}

export function useDocuments() {
  const qc = useQueryClient();

  const list = useQuery<DocumentItem[]>({
    queryKey: ["documents"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) throw new Error("文档列表加载失败，请刷新页面重试");
      return res.json();
    },
  });

  const upload = useMutation({
    mutationFn: async ({ formData, signal }: { formData: FormData; signal?: AbortSignal }) => {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
        signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(friendlyError(res.status, text));
      }
      return res.json();
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
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

  return { list, upload, remove };
}
