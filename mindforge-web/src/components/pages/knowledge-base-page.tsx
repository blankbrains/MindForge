import { useState, useRef } from "react";
import { useDocuments } from "@/hooks/use-documents";
import { useStats } from "@/hooks/use-stats";
import { EmptyState } from "@/components/shared/empty-state";
import { LoadingSkeleton } from "@/components/shared/loading-skeleton";
import { FileText, Upload, X, Database, HardDrive, Eye, Loader2, Trash2 } from "lucide-react";
import { API_BASE } from "@/lib/constants";

export function KnowledgeBasePage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const { list: documents, upload, remove } = useDocuments();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useRaptor, setUseRaptor] = useState(false);
  const [useGraphrag, setUseGraphrag] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Document content modal
  const [viewingDoc, setViewingDoc] = useState<{ doc_id: string; filename: string } | null>(null);
  const [docContent, setDocContent] = useState<string>("");
  const [docLoading, setDocLoading] = useState(false);
  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<{ doc_id: string; filename: string } | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Upload cancel confirmation
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [uploadAbortController, setUploadAbortController] = useState<AbortController | null>(null);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleteError(null);
    try {
      await remove.mutateAsync(deleteTarget.doc_id);
      setDeleteTarget(null);
      if (viewingDoc?.doc_id === deleteTarget.doc_id) {
        setViewingDoc(null);
        setDocContent("");
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "删除失败");
    }
  };

  const handleUpload = () => {
    if (!selectedFile) return;
    setUploadError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("use_raptor", String(useRaptor));
    formData.append("use_graphrag", String(useGraphrag));
    const controller = new AbortController();
    setUploadAbortController(controller);
    upload.mutate(
      { formData, signal: controller.signal },
      {
        onSuccess: () => { setUploadOpen(false); setSelectedFile(null); setUseRaptor(false); setUseGraphrag(false); setUploadAbortController(null); },
        onError: (err) => { if (err instanceof Error && err.name !== "AbortError") setUploadError(err.message || "上传失败"); setUploadAbortController(null); },
      },
    );
  };

  const handleCancelUpload = () => {
    if (upload.isPending) {
      setCancelConfirmOpen(true);
    } else {
      setUploadOpen(false);
      setSelectedFile(null);
    }
  };

  const confirmCancelUpload = () => {
    uploadAbortController?.abort();
    setCancelConfirmOpen(false);
    setUploadOpen(false);
    setSelectedFile(null);
    setUploadAbortController(null);
  };

  const handleViewDocument = async (docId: string, filename: string) => {
    setViewingDoc({ doc_id: docId, filename });
    setDocLoading(true);
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}/content`);
      if (res.ok) {
        const data = await res.json();
        setDocContent(data.content || "（无内容）");
      } else {
        setDocContent("加载失败");
      }
    } catch {
      setDocContent("加载失败");
    } finally {
      setDocLoading(false);
    }
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

      {documents.isLoading ? <LoadingSkeleton variant="card" count={4} /> : documents.isError ? (
        <EmptyState icon={<XCircle className="h-12 w-12" />} title="加载失败" description="无法获取文档列表，请检查后端服务是否运行" />
      ) : documents.data && documents.data.length > 0 ? (
        <div className="space-y-2">
          {documents.data.map((doc) => (
            <div
              key={doc.doc_id}
              className="flex items-center gap-2 rounded-xl border border-border bg-surface px-5 py-4 transition-colors hover:border-primary/30 hover:shadow-sm group"
            >
              <button
                type="button"
                onClick={() => handleViewDocument(doc.doc_id, doc.filename)}
                className="flex flex-1 items-center gap-4 text-left cursor-pointer"
              >
                <FileText className="h-5 w-5 text-text-muted shrink-0" />
                <div className="flex-1 min-w-0"><p className="font-medium truncate">{doc.filename}</p><p className="text-xs text-text-muted">{doc.chunk_count} 块 · {doc.status}</p></div>
                <Eye className="h-4 w-4 text-text-muted opacity-50 group-hover:opacity-100 transition-opacity" />
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setDeleteTarget({ doc_id: doc.doc_id, filename: doc.filename }); }}
                disabled={remove.isPending}
                className="shrink-0 rounded-lg p-1.5 text-text-muted hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950 transition-colors opacity-0 group-hover:opacity-100"
                title="删除文档"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState icon={<FileText className="h-12 w-12" />} title="暂无文档" description="上传 PDF、DOCX、Markdown 等文件到知识库" action={<button type="button" onClick={() => setUploadOpen(true)} className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark transition-colors">上传第一篇文档</button>} />
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setDeleteTarget(null)}>
          <div className="w-full max-w-sm rounded-2xl border border-border bg-surface p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">确认删除</h3>
              <button type="button" onClick={() => setDeleteTarget(null)} className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"><X className="h-5 w-5" /></button>
            </div>
            <p className="text-sm text-text-muted mb-2">将永久删除文档及其所有索引数据：</p>
            <p className="text-sm font-medium text-text mb-4 truncate">{deleteTarget.filename}</p>
            {deleteError && <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">{deleteError}</div>}
            <div className="flex gap-3">
              <button type="button" onClick={() => setDeleteTarget(null)} className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt transition-colors">取消</button>
              <button type="button" onClick={handleDelete} disabled={remove.isPending} className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50 transition-colors">{remove.isPending ? "删除中…" : "确认删除"}</button>
            </div>
          </div>
        </div>
      )}

      {/* Cancel Upload Confirmation */}
      {cancelConfirmOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setCancelConfirmOpen(false)}>
          <div className="w-full max-w-sm rounded-2xl border border-border bg-surface p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold mb-2">确认停止上传</h3>
            <p className="text-sm text-text-muted mb-4">文档正在索引中，停止后已上传的部分可能不完整。确定要停止吗？</p>
            <div className="flex gap-3">
              <button type="button" onClick={() => setCancelConfirmOpen(false)} className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt transition-colors">继续上传</button>
              <button type="button" onClick={confirmCancelUpload} className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 transition-colors">停止上传</button>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={(e) => { if (e.target === e.currentTarget && !upload.isPending) handleCancelUpload(); }}>
          <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">上传文档</h3>
              <button type="button" onClick={handleCancelUpload} className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"><X className="h-5 w-5" /></button>
            </div>
            <div className="mt-5 space-y-4">
              <div>
                <label className="block text-sm font-medium text-text mb-1.5">选择文件</label>
                <input ref={fileRef} type="file" accept=".pdf,.docx,.md,.txt,.html" disabled={upload.isPending} onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-primary file:px-3 file:py-1 file:text-sm file:text-white hover:file:bg-primary-dark focus:ring-2 focus:ring-primary/20 focus:border-primary/50 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed" />
                {selectedFile && <p className="mt-1 text-xs text-text-muted">已选择: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</p>}
              </div>
              <div className="flex items-center gap-5">
                <label className="flex items-center gap-2 text-sm cursor-pointer"><input type="checkbox" checked={useRaptor} onChange={(e) => setUseRaptor(e.target.checked)} className="rounded border-border" /><span className="font-medium">RAPTOR</span><span className="text-xs text-text-muted">层次索引</span></label>
                <label className="flex items-center gap-2 text-sm cursor-pointer"><input type="checkbox" checked={useGraphrag} onChange={(e) => setUseGraphrag(e.target.checked)} className="rounded border-border" /><span className="font-medium">GraphRAG</span><span className="text-xs text-text-muted">图谱索引</span></label>
              </div>
              {uploadError && <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">{uploadError}</div>}
              <button type="button" onClick={handleUpload} disabled={!selectedFile || upload.isPending} className="w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-white hover:bg-primary-dark disabled:opacity-50 transition-colors">{upload.isPending ? "索引中…" : "开始索引"}</button>
            </div>
          </div>
        </div>
      )}

      {/* Document Content Modal */}
      {viewingDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={(e) => { if (e.target === e.currentTarget) { setViewingDoc(null); setDocContent(""); } }}>
          <div className="w-full max-w-3xl max-h-[80vh] rounded-2xl border border-border bg-surface p-6 shadow-2xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold truncate">{viewingDoc.filename}</h3>
              <button type="button" onClick={() => { setViewingDoc(null); setDocContent(""); }} className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"><X className="h-5 w-5" /></button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {docLoading ? (
                <div className="flex items-center justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-text-muted" /></div>
              ) : (
                <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans">{docContent}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
