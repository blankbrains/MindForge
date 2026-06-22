import { createLazyRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useDocuments } from "@/hooks/use-documents";
import { useStats } from "@/hooks/use-stats";
import { EmptyState } from "@/components/shared/empty-state";
import { LoadingSkeleton } from "@/components/shared/loading-skeleton";
import { FileText, Upload, Trash2, X } from "lucide-react";

function KnowledgeBasePage() {
  const { data: stats } = useStats();
  const { list: documents, upload, remove } = useDocuments();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [filePath, setFilePath] = useState("");
  const [useRaptor, setUseRaptor] = useState(false);
  const [useGraphrag, setUseGraphrag] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const handleUpload = () => {
    if (!filePath.trim()) return;
    upload.mutate(
      { file_path: filePath.trim(), use_raptor: useRaptor, use_graphrag: useGraphrag },
      { onSuccess: () => { setUploadOpen(false); setFilePath(""); } },
    );
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">知识库</h1>
          <p className="mt-1 text-text-muted">
            已索引 {stats?.documents_indexed ?? 0} 个文档
          </p>
        </div>
        <button
          type="button"
          onClick={() => setUploadOpen(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-dark"
        >
          <Upload className="h-4 w-4" />
          上传文档
        </button>
      </div>

      {/* Document list placeholder */}
      {documents.isLoading ? (
        <LoadingSkeleton variant="card" count={4} />
      ) : (
        <EmptyState
          icon={<FileText className="h-12 w-12" />}
          title="暂无文档"
          description="上传 PDF、DOCX、Markdown 等文件到知识库"
          action={
            <button
              type="button"
              onClick={() => setUploadOpen(true)}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark"
            >
              上传第一篇文档
            </button>
          }
        />
      )}

      {/* Upload dialog */}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">上传文档</h3>
              <button
                type="button"
                onClick={() => setUploadOpen(false)}
                className="rounded-lg p-1 text-text-muted hover:bg-surface-alt"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text">
                  文件路径
                </label>
                <input
                  type="text"
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  placeholder="例如: data/my-report.pdf"
                  className="mt-1 w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary/50 focus:outline-none"
                />
              </div>

              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useRaptor}
                    onChange={(e) => setUseRaptor(e.target.checked)}
                    className="rounded border-border text-primary focus:ring-primary"
                  />
                  RAPTOR 索引
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={useGraphrag}
                    onChange={(e) => setUseGraphrag(e.target.checked)}
                    className="rounded border-border text-primary focus:ring-primary"
                  />
                  GraphRAG 索引
                </label>
              </div>

              <button
                type="button"
                onClick={handleUpload}
                disabled={!filePath.trim() || upload.isPending}
                className="w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-white hover:bg-primary-dark disabled:opacity-50"
              >
                {upload.isPending ? "索引中…" : "开始索引"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const Route = createLazyRoute("/knowledge-base")({
  component: KnowledgeBasePage,
});
