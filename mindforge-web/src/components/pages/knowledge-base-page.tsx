import { useEffect, useRef, useState } from "react";
import { useDocuments } from "@/hooks/use-documents";
import { useStats } from "@/hooks/use-stats";
import { EmptyState } from "@/components/shared/empty-state";
import { LoadingSkeleton } from "@/components/shared/loading-skeleton";
import { Modal } from "@/components/shared/modal";
import { HelpTooltip, Tooltip } from "@/components/shared/tooltip";
import {
  Database,
  Eye,
  FileText,
  HardDrive,
  Layers3,
  Loader2,
  Network,
  Trash2,
  Upload,
  X,
  XCircle,
} from "lucide-react";
import { API_BASE } from "@/lib/constants";

const INDEX_STAGE_LABELS: Record<string, string> = {
  detecting: "正在检测文档结构",
  ocr: "正在识别扫描页",
  layout: "正在分析版面",
  table: "正在提取表格",
  queued: "等待服务器处理",
  parsing: "正在解析文档",
  chunking: "正在生成知识块",
  embedding: "正在生成向量",
  vector_store: "正在写入向量库",
  bm25: "正在构建关键词索引",
  raptor: "正在构建层次索引",
  graphrag: "正在构建知识图谱",
  completed: "索引完成",
  failed: "索引失败",
  cancelled: "已取消",
};

export function KnowledgeBasePage() {
  const { data: stats, isLoading: statsLoading } = useStats();
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const {
    list: documents,
    job,
    upload,
    cancelJob,
    remove,
    setEnabled,
  } = useDocuments(activeJobId);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [useRaptor, setUseRaptor] = useState(false);
  const [useGraphrag, setUseGraphrag] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Document content modal
  const [viewingDoc, setViewingDoc] = useState<{
    doc_id: string;
    filename: string;
  } | null>(null);
  const [docContent, setDocContent] = useState<string>("");
  const [docLoading, setDocLoading] = useState(false);
  const documentRequestRef = useRef<AbortController | null>(null);
  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<{
    doc_id: string;
    filename: string;
  } | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [availabilityError, setAvailabilityError] = useState<string | null>(
    null,
  );
  // Upload cancel confirmation
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [uploadAbortController, setUploadAbortController] =
    useState<AbortController | null>(null);
  const maxUploadMb = stats?.max_upload_mb ?? 200;
  const maxPdfPages = stats?.max_pdf_pages ?? 500;
  const activeJob = job.data;
  const indexing =
    activeJob?.status === "queued" || activeJob?.status === "running";
  const uploadBusy = upload.isPending || indexing || cancelJob.isPending;
  const uploadDialogOpen =
    uploadOpen &&
    activeJob?.status !== "completed" &&
    activeJob?.status !== "cancelled";
  const activeJobError =
    activeJob?.status === "failed" ? activeJob.error || "文档索引失败" : null;
  const cancelJobError =
    cancelJob.error instanceof Error
      ? cancelJob.error.message
      : cancelJob.error
        ? "取消索引失败，请稍后重试"
        : null;

  useEffect(() => {
    return () => documentRequestRef.current?.abort();
  }, []);

  const closeDocument = () => {
    documentRequestRef.current?.abort();
    documentRequestRef.current = null;
    setViewingDoc(null);
    setDocContent("");
    setDocLoading(false);
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleteError(null);
    try {
      await remove.mutateAsync(deleteTarget.doc_id);
      setDeleteTarget(null);
      if (viewingDoc?.doc_id === deleteTarget.doc_id) {
        closeDocument();
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "删除失败");
    }
  };

  const handleUpload = () => {
    if (!selectedFile) return;
    if (selectedFile.size > maxUploadMb * 1024 * 1024) {
      setUploadError(`文件过大，最大支持 ${maxUploadMb}MB`);
      return;
    }
    setUploadError(null);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("use_raptor", String(useRaptor));
    formData.append("use_graphrag", String(useGraphrag));
    const controller = new AbortController();
    setUploadAbortController(controller);
    upload.mutate(
      {
        formData,
        signal: controller.signal,
        onUploadProgress: ({ percent }) => setUploadProgress(percent),
      },
      {
        onSuccess: (createdJob) => {
          setUploadProgress(100);
          setActiveJobId(createdJob.job_id);
          setUploadAbortController(null);
        },
        onError: (err) => {
          setUploadProgress(null);
          if (err instanceof Error && err.name !== "AbortError") {
            setUploadError(err.message || "上传失败");
          }
          setUploadAbortController(null);
        },
      },
    );
  };

  const openUploadDialog = () => {
    cancelJob.reset?.();
    setSelectedFile(null);
    setUseRaptor(false);
    setUseGraphrag(false);
    setUploadError(null);
    setUploadProgress(null);
    setActiveJobId(null);
    setUploadOpen(true);
  };

  const handleCancelUpload = () => {
    if (uploadBusy) {
      setCancelConfirmOpen(true);
    } else {
      setUploadOpen(false);
      setSelectedFile(null);
      setUploadProgress(null);
      setActiveJobId(null);
    }
  };

  const handleAvailabilityChange = async (
    docId: string,
    filename: string,
    enabled: boolean,
  ) => {
    setAvailabilityError(null);
    try {
      await setEnabled.mutateAsync({ docId, enabled });
    } catch (error) {
      const action = enabled ? "启用" : "禁用";
      setAvailabilityError(
        error instanceof Error
          ? error.message
          : `${action}文档 ${filename} 失败，请稍后重试`,
      );
    }
  };

  const confirmCancelUpload = () => {
    if (upload.isPending) {
      uploadAbortController?.abort();
    } else if (activeJobId && indexing) {
      cancelJob.reset?.();
      cancelJob.mutate(activeJobId);
    }
    setCancelConfirmOpen(false);
    setUploadAbortController(null);
  };

  const handleViewDocument = async (docId: string, filename: string) => {
    documentRequestRef.current?.abort();
    const controller = new AbortController();
    documentRequestRef.current = controller;
    setViewingDoc({ doc_id: docId, filename });
    setDocContent("");
    setDocLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/documents/${docId}/content?include_chunks=false`,
        { signal: controller.signal },
      );
      if (documentRequestRef.current !== controller) return;
      if (res.ok) {
        const data = await res.json();
        setDocContent(data.content || "（无内容）");
      } else {
        setDocContent("加载失败");
      }
    } catch (error) {
      if (
        documentRequestRef.current === controller &&
        (!(error instanceof Error) || error.name !== "AbortError")
      ) {
        setDocContent("加载失败");
      }
    } finally {
      if (documentRequestRef.current === controller) {
        documentRequestRef.current = null;
        setDocLoading(false);
      }
    }
  };

  return (
    <div className="mx-auto w-full min-w-0 max-w-5xl space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold tracking-tight">知识库</h1>
          <p className="mt-1 text-text-muted">
            已索引 {statsLoading ? "…" : (stats?.documents_indexed ?? 0)} 个文档
          </p>
        </div>
        <button
          type="button"
          onClick={openUploadDialog}
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-dark transition-colors"
        >
          <Upload className="h-4 w-4" /> 上传文档
        </button>
      </div>

      {!statsLoading && stats && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-text-muted">已索引文档</p>
              <p className="text-xl font-bold">{stats.documents_indexed}</p>
            </div>
          </div>
          <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <HardDrive className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-text-muted">向量数据库</p>
              <p className="text-xl font-bold">
                {stats.qdrant_connected ? "已连接" : "未连接"}
              </p>
            </div>
          </div>
        </div>
      )}

      {documents.isLoading ? (
        <LoadingSkeleton variant="card" count={4} />
      ) : documents.isError ? (
        <EmptyState
          icon={<XCircle className="h-12 w-12" />}
          title="加载失败"
          description="无法获取文档列表，请检查后端服务是否运行"
        />
      ) : documents.data && documents.data.length > 0 ? (
        <div className="space-y-2">
          {availabilityError && (
            <div
              role="alert"
              className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
            >
              {availabilityError}
            </div>
          )}
          {documents.data.map((doc) => (
            <div
              key={doc.doc_id}
              className={`group flex items-center gap-2 rounded-lg border px-5 py-4 transition-colors ${
                doc.enabled
                  ? "border-border bg-surface hover:border-primary/30 hover:shadow-sm"
                  : "border-dashed border-border bg-surface-alt/60"
              }`}
            >
              <button
                type="button"
                onClick={() => handleViewDocument(doc.doc_id, doc.filename)}
                className="flex min-w-0 flex-1 items-center gap-4 text-left cursor-pointer"
              >
                <FileText className="h-5 w-5 text-text-muted shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{doc.filename}</p>
                  <p className="text-xs text-text-muted">
                    {doc.chunk_count} 块 · {doc.status}
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {!doc.enabled && (
                      <span className="inline-flex items-center rounded-md border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[11px] font-medium text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
                        已禁用
                      </span>
                    )}
                    <span className="inline-flex items-center rounded-md border border-border bg-surface-alt px-1.5 py-0.5 text-[11px] text-text-muted">
                      基础索引
                    </span>
                    {doc.use_raptor && (
                      <span className="inline-flex items-center gap-1 rounded-md border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-[11px] text-primary">
                        <Layers3 className="h-3 w-3" aria-hidden="true" />
                        RAPTOR 层次索引
                      </span>
                    )}
                    {doc.use_graphrag && (
                      <span className="inline-flex items-center gap-1 rounded-md border border-accent/40 bg-accent/10 px-1.5 py-0.5 text-[11px] text-text">
                        <Network className="h-3 w-3" aria-hidden="true" />
                        GraphRAG 图谱索引
                      </span>
                    )}
                  </div>
                </div>
                <Eye className="h-4 w-4 text-text-muted opacity-50 group-hover:opacity-100 transition-opacity" />
              </button>
              <div className="flex shrink-0 items-center gap-2">
                <span className="hidden text-xs text-text-muted sm:inline">
                  检索
                </span>
                <Tooltip
                  content={
                    doc.enabled
                      ? "关闭后文档仍保留，但不会参与后续知识库检索。"
                      : "开启后文档会重新参与后续知识库检索。"
                  }
                  side="top"
                >
                  <button
                    type="button"
                    role="switch"
                    aria-checked={doc.enabled}
                    aria-label={`${doc.enabled ? "禁用" : "启用"}文档 ${doc.filename}`}
                    disabled={
                      setEnabled.isPending &&
                      setEnabled.variables?.docId === doc.doc_id
                    }
                    onClick={() =>
                      void handleAvailabilityChange(
                        doc.doc_id,
                        doc.filename,
                        !doc.enabled,
                      )
                    }
                    className={`relative inline-flex h-6 w-10 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-60 ${
                      doc.enabled ? "bg-emerald-600" : "bg-border"
                    }`}
                  >
                    <span
                      aria-hidden="true"
                      className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                        doc.enabled ? "translate-x-5" : "translate-x-1"
                      }`}
                    />
                  </button>
                </Tooltip>
              </div>
              <Tooltip
                content="永久删除文档正文、知识块、向量索引和相关图谱数据。"
                side="left"
              >
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteTarget({
                      doc_id: doc.doc_id,
                      filename: doc.filename,
                    });
                  }}
                  disabled={remove.isPending}
                  className="shrink-0 rounded-lg p-1.5 text-text-muted opacity-100 transition-colors hover:bg-red-50 hover:text-red-500 disabled:opacity-50 sm:opacity-0 sm:group-hover:opacity-100 sm:focus:opacity-100 dark:hover:bg-red-950"
                  aria-label={`删除文档 ${doc.filename}`}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </Tooltip>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<FileText className="h-12 w-12" />}
          title="暂无文档"
          description="上传 PDF、DOCX、Markdown 等文件到知识库"
          action={
            <button
              type="button"
              onClick={openUploadDialog}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dark transition-colors"
            >
              上传第一篇文档
            </button>
          }
        />
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <Modal
          titleId="delete-document-title"
          descriptionId="delete-document-description"
          onClose={() => {
            if (!remove.isPending) setDeleteTarget(null);
          }}
          closeOnBackdrop={!remove.isPending}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 id="delete-document-title" className="text-lg font-semibold">
              确认删除
            </h2>
            <Tooltip content="关闭删除确认" side="left">
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                disabled={remove.isPending}
                aria-label="关闭删除确认"
                className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors disabled:opacity-50"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
          <p
            id="delete-document-description"
            className="text-sm text-text-muted mb-2"
          >
            将永久删除文档及其所有索引数据：
          </p>
          <p className="text-sm font-medium text-text mb-4 truncate">
            {deleteTarget.filename}
          </p>
          {deleteError && (
            <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
              {deleteError}
            </div>
          )}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setDeleteTarget(null)}
              className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt transition-colors"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleDelete}
              disabled={remove.isPending}
              className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50 transition-colors"
            >
              {remove.isPending ? "删除中…" : "确认删除"}
            </button>
          </div>
        </Modal>
      )}

      {/* Upload Modal */}
      {uploadDialogOpen && (
        <Modal
          titleId="upload-document-title"
          onClose={handleCancelUpload}
          closeOnBackdrop={!uploadBusy}
          className="max-w-md"
        >
          <div className="flex items-center justify-between">
            <h2 id="upload-document-title" className="text-lg font-semibold">
              上传文档
            </h2>
            <Tooltip content="关闭上传窗口" side="left">
              <button
                type="button"
                onClick={handleCancelUpload}
                aria-label="关闭上传窗口"
                className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
          <div className="mt-5 space-y-4">
            {!activeJobId && !upload.isPending && (
              <div>
                <label
                  htmlFor="knowledge-upload-file"
                  className="block text-sm font-medium text-text mb-1.5"
                >
                  选择文件
                </label>
                <input
                  id="knowledge-upload-file"
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.docx,.md,.txt,.html,.htm"
                  disabled={upload.isPending}
                  onChange={(e) => {
                    const nextFile = e.target.files?.[0] || null;
                    setSelectedFile(nextFile);
                    setUploadError(
                      nextFile && nextFile.size > maxUploadMb * 1024 * 1024
                        ? `文件过大，最大支持 ${maxUploadMb}MB`
                        : null,
                    );
                  }}
                  className="w-full rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-primary file:px-3 file:py-1 file:text-sm file:text-white hover:file:bg-primary-dark focus:ring-2 focus:ring-primary/20 focus:border-primary/50 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                />
                {selectedFile && (
                  <p className="mt-1 text-xs text-text-muted">
                    已选择: {selectedFile.name} (
                    {(selectedFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
                <p className="mt-1 text-xs text-text-muted">
                  单个文件最大 {maxUploadMb}MB，PDF 最多 {maxPdfPages} 页
                </p>
              </div>
            )}
            {!activeJobId && !upload.isPending && (
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
                <div className="flex items-center gap-1">
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={useRaptor}
                      onChange={(e) => setUseRaptor(e.target.checked)}
                      className="rounded border-border"
                    />
                    <span className="font-medium">RAPTOR</span>
                    <span className="text-xs text-text-muted">层次索引</span>
                  </label>
                  <HelpTooltip
                    label="RAPTOR 说明"
                    content="把文档知识块递归汇总为多层摘要，适合长文档和跨章节问题，但会增加索引时间与存储。"
                  />
                </div>
                <div className="flex items-center gap-1">
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={useGraphrag}
                      onChange={(e) => setUseGraphrag(e.target.checked)}
                      className="rounded border-border"
                    />
                    <span className="font-medium">GraphRAG</span>
                    <span className="text-xs text-text-muted">图谱索引</span>
                  </label>
                  <HelpTooltip
                    label="GraphRAG 说明"
                    content="提取实体与关系并构建知识图谱，适合关系推理和跨文档关联，也会增加索引耗时。"
                  />
                </div>
              </div>
            )}
            {upload.isPending && !activeJobId && (
              <div className="space-y-3" aria-live="polite">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 shrink-0 text-text-muted" />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {selectedFile?.name}
                    </p>
                    <p className="text-xs text-text-muted">正在上传到服务器</p>
                  </div>
                </div>
                <div
                  className="h-2 overflow-hidden bg-surface-alt"
                  role="progressbar"
                  aria-label="文档上传进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={
                    uploadProgress === null
                      ? undefined
                      : Math.round(uploadProgress)
                  }
                >
                  <div
                    className={
                      uploadProgress === null
                        ? "h-full w-1/3 animate-pulse bg-primary"
                        : "h-full bg-primary transition-[width] duration-200"
                    }
                    style={
                      uploadProgress === null
                        ? undefined
                        : {
                            width: `${Math.max(
                              0,
                              Math.min(100, uploadProgress),
                            )}%`,
                          }
                    }
                  />
                </div>
                <p className="text-right text-xs text-text-muted">
                  {uploadProgress === null
                    ? "正在传输…"
                    : `${Math.round(uploadProgress)}%`}
                </p>
              </div>
            )}
            {activeJobId && (
              <div className="space-y-3" aria-live="polite">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 shrink-0 text-text-muted" />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">
                      {activeJob?.filename || selectedFile?.name}
                    </p>
                    <p className="text-xs text-text-muted">
                      {job.isError
                        ? "进度获取失败，正在重试"
                        : activeJob?.cancel_requested
                          ? "正在取消"
                          : INDEX_STAGE_LABELS[activeJob?.stage || ""] ||
                            activeJob?.stage ||
                            "等待处理"}
                    </p>
                  </div>
                </div>
                <div
                  className="h-2 overflow-hidden bg-surface-alt"
                  role="progressbar"
                  aria-label="当前文档索引进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(activeJob?.progress ?? 0)}
                >
                  <div
                    className="h-full bg-primary transition-[width] duration-300"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, activeJob?.progress ?? 0),
                      )}%`,
                    }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-text-muted">
                  <span>
                    {Math.round(activeJob?.progress ?? 0)}%
                    {(activeJob?.chunk_count ?? 0) > 0
                      ? ` · ${activeJob?.chunk_count} 块`
                      : ""}
                  </span>
                  <span>
                    {activeJob?.timings.total !== undefined
                      ? `${activeJob.timings.total.toFixed(1)} 秒`
                      : ""}
                  </span>
                </div>
              </div>
            )}
            {(uploadError || activeJobError || cancelJobError) && (
              <div
                role="alert"
                className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
              >
                {uploadError || activeJobError || cancelJobError}
              </div>
            )}
            {!activeJobId && !upload.isPending && (
              <button
                type="button"
                onClick={handleUpload}
                disabled={!selectedFile}
                className="w-full rounded-lg bg-primary py-2.5 text-sm font-medium text-white hover:bg-primary-dark disabled:opacity-50 transition-colors"
              >
                开始索引
              </button>
            )}
            {upload.isPending && (
              <button
                type="button"
                onClick={() => setCancelConfirmOpen(true)}
                className="w-full rounded-lg border border-red-300 py-2.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 dark:border-red-800 dark:text-red-300 dark:hover:bg-red-950"
              >
                取消上传
              </button>
            )}
            {indexing && (
              <button
                type="button"
                onClick={() => setCancelConfirmOpen(true)}
                disabled={cancelJob.isPending || activeJob?.cancel_requested}
                className="w-full rounded-lg border border-red-300 py-2.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-50 dark:border-red-800 dark:text-red-300 dark:hover:bg-red-950"
              >
                {cancelJob.isPending || activeJob?.cancel_requested
                  ? "取消中…"
                  : "取消本次索引"}
              </button>
            )}
          </div>
        </Modal>
      )}

      {/* Render after the upload dialog so this remains the active modal. */}
      {cancelConfirmOpen && (
        <Modal
          titleId="cancel-upload-title"
          descriptionId="cancel-upload-description"
          onClose={() => setCancelConfirmOpen(false)}
        >
          <h2 id="cancel-upload-title" className="text-lg font-semibold mb-2">
            {upload.isPending ? "确认取消上传" : "确认取消索引"}
          </h2>
          <p
            id="cancel-upload-description"
            className="text-sm text-text-muted mb-4"
          >
            {upload.isPending
              ? "文件尚未上传完成。确定要中止本次上传吗？"
              : "服务器会在当前阶段结束后停止并回滚本次索引。确定要取消吗？"}
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setCancelConfirmOpen(false)}
              className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-muted hover:bg-surface-alt transition-colors"
            >
              {upload.isPending ? "继续上传" : "继续索引"}
            </button>
            <button
              type="button"
              onClick={confirmCancelUpload}
              className="flex-1 rounded-lg bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 transition-colors"
            >
              {upload.isPending ? "停止上传" : "取消索引"}
            </button>
          </div>
        </Modal>
      )}

      {/* Document Content Modal */}
      {viewingDoc && (
        <Modal
          titleId="document-content-title"
          onClose={closeDocument}
          className="flex max-h-[80vh] max-w-3xl flex-col overflow-hidden"
        >
          <div className="flex items-center justify-between mb-4">
            <h2
              id="document-content-title"
              className="text-lg font-semibold truncate"
            >
              {viewingDoc.filename}
            </h2>
            <Tooltip content="关闭文档内容" side="left">
              <button
                type="button"
                onClick={closeDocument}
                aria-label="关闭文档内容"
                className="rounded-lg p-1 text-text-muted hover:bg-surface-alt transition-colors"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
          <div className="flex-1 overflow-y-auto">
            {docLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
              </div>
            ) : (
              <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans">
                {docContent}
              </pre>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}
