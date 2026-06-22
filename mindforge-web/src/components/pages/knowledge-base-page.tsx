import { useState } from "react";
import { useDocuments } from "@/hooks/use-documents";
import { useStats } from "@/hooks/use-stats";
import { EmptyState } from "@/components/shared/empty-state";
import { LoadingSkeleton } from "@/components/shared/loading-skeleton";
import { FileText, Upload, X, Database, HardDrive } from "lucide-react";

export function KnowledgeBasePage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { list: documents, upload } = useDocuments();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [filePath, setFilePath] = useState("");
  const [useRaptor, setUseRaptor] = useState(false);
  const [useGraphrag, setUseGraphrag] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleUpload = () => {
    if (!filePath.trim()) return;
    setUploadError(null);
    upload.mutate(
      { file_path: filePath.trim(), use_raptor: useRaptor, use_graphrag: useGraphrag },
      {
        onSuccess: () => { setUploadOpen(false); setFilePath(""); setUseRaptor(false); setUseGraphrag(false); },
        onError: (err) => setUploadError(err instanceof Error ? err.message : "上传失败"),
      },
    );
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">知识库</h1>
          <p className="mt-1 text-text-muted">已索引 {statsLoading ? "…" : stats?.documents_indexed ?? 0} 个文档</p>
        </div>
        <button type="button" onClick={() => setUploadOpen(true)} className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-dark transition-colors">
          <Upload className="h-4 w-4" /> 上传文档
        </button>
      </div>
      {!statsLoading && stats && (
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary"><Database className="h-5 w-5" /></div>
            <div><p className="text-sm text-text-muted">已索引文档</p><p className="text-xl font-bold">{stats.documents_indexed}</p></div>
          </div>
          <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent"><HardDrive className="h-5 w-5" /></div>
            <div><p className="text-sm text-text-muted">向量数据库</p><p className="text-xl font-bold">{stats.qdrant_url ? "已连接" : "未连接"}</p></div>
          </div>
        </div>
      )}
      {documents.isLoading ? <LoadingSkeleton variant="card" count={4} /> : documents.data && documents.data.length > 0 ? (
        <div className="space-y-2">
          {documents.data.map((doc) => (
            <div key={doc.doc_id} className="flex items-center gap-4 rounded-xl border border-border bg-surface px-5 py-4">
              <FileText className="h-5 w-5 text-text-muted shrink-0" />
              <div className="flex-1 min-w-0"><p className="font-medium truncate">{doc.filename}</p><p className="text-xs text-text-muted">{doc.chunk_count} 块 · {doc.status}</p></div>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState icon={<FileText className="h-12 w-12" />} title="暂无文档" description="上传 PDF、DOCX、Markdown 等文件到知识库" action={<button type="button" onClick={() => setUploadOpen(true)} className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark transition-colors">上传第一篇文档</button>} />
      )}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={(e) => { if (e.target === e.currentTarget) setUploadOpen(false); }}>
          <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">上传文档</h3>
              <button type="button" onClick={() => setUploadOpen(false)} className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"><X className="h-5 w-5" /></button>
            </div>
            <div className="mt-5 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text mb-1.5">文件路径</label>
                <input type="text" value={filePath} onChange={(e) => setFilePath(e.target.value)} placeholder="例如: data/my-report.pdf" className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary/50 focus:outline-none" onKeyDown={(e) => { if (e.key === "Enter") handleUpload(); }} />
                <p className="mt-1 text-xs text-text-muted">后端 data/ 目录下的文件路径</p>
              </div>
              <div className="flex items-center gap-5">
                <label className="flex items-center gap-2 text-sm cursor-pointer"><input type="checkbox" checked={useRaptor} onChange={(e) => setUseRaptor(e.target.checked)} className="rounded border-border" /><span className="font-medium">RAPTOR</span><span className="text-xs text-text-muted">层次索引</span></label>
                <label className="flex items-center gap-2 text-sm cursor-pointer"><input type="checkbox" checked={useGraphrag} onChange={(e) => setUseGraphrag(e.target.checked)} className="rounded border-border" /><span className="font-medium">GraphRAG</span><span className="text-xs text-text-muted">图谱索引</span></label>
              </div>
              {uploadError && <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">{uploadError}</div>}
              <button type="button" onClick={handleUpload} disabled={!filePath.trim() || upload.isPending} className="w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-white hover:bg-primary-dark disabled:opacity-50 transition-colors">{upload.isPending ? "索引中…" : "开始索引"}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
