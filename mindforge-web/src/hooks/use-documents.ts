import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { DocumentItem, IndexResponse } from "@/types/document";

export function useDocuments() {
  const qc = useQueryClient();

  const list = useQuery<DocumentItem[]>({
    queryKey: ["documents"],
    queryFn: () => api.get("/stats").then((s) => []), // placeholder until backend adds list endpoint
  });

  const upload = useMutation({
    mutationFn: (data: {
      file_path?: string;
      file_url?: string;
      use_raptor?: boolean;
      use_graphrag?: boolean;
    }) => api.post<IndexResponse>("/index", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const remove = useMutation({
    mutationFn: (docId: string) => api.delete(`/documents/${docId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["stats"] });
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  return { list, upload, remove };
}
