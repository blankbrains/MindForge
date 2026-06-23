import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { API_BASE } from "@/lib/constants";
import type { DocumentItem } from "@/types/document";

export function useDocuments() {
  const qc = useQueryClient();

  const list = useQuery<DocumentItem[]>({
    queryKey: ["documents"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) throw new Error("Failed to fetch documents");
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
        throw new Error(text || "Upload failed");
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
      if (!res.ok) throw new Error("Delete failed");
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  return { list, upload, remove };
}
