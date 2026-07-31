import { Database, Globe2, ScanSearch } from "lucide-react";
import {
  useSettingsStore,
  type EmbeddingProvider,
} from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

export function RetrievalSettingsPanel() {
  const state = useSettingsStore();

  return (
    <section
      className="space-y-6 border border-border bg-surface p-5 sm:p-6"
      role="tabpanel"
      aria-label="检索配置"
    >
      <div className="grid grid-cols-1 gap-px border border-border bg-border sm:grid-cols-2">
        <StatusItem
          icon={ScanSearch}
          label="重排序模型"
          value={
            state.rerankerAvailable
              ? "运行正常"
              : state.rerankerLoadFailed
                ? "模型加载失败"
              : state.rerankerConfigured
                ? "等待模型加载"
                : "未启用"
          }
          detail={
            state.rerankerAvailable
              ? "CrossEncoder 精排已启用"
              : state.rerankerLoadFailed
                ? "继续使用基础混合检索；请检查模型文件、依赖和设备配置"
                : state.rerankerConfigured
                  ? "模型名称已设置，将在服务加载后启用"
                  : "未设置重排序模型"
          }
          tone={
            state.rerankerAvailable
              ? "success"
              : state.rerankerConfigured
                ? "warning"
                : "muted"
          }
        />
        <StatusItem
          icon={Globe2}
          label="联网搜索"
          value={state.tavilyConfigured ? "Tavily 可用" : "未配置 API Key"}
          detail={
            state.tavilyConfigured
              ? "Agent 可以调用联网搜索"
              : "当前不会调用联网搜索；配置 TAVILY_API_KEY 后启用"
          }
          tone={state.tavilyConfigured ? "success" : "warning"}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium">Embedding</span>
          <select
            value={state.embeddingProvider}
            onChange={(event) =>
              state.setEmbeddingProvider(
                event.target.value as EmbeddingProvider,
              )
            }
            className={inputClassName}
          >
            <option value="bge">本地 BGE</option>
            <option value="openai">OpenAI Embedding</option>
          </select>
          <span className="mt-1 block text-xs leading-5 text-text-muted">
            已有知识库时不能直接切换，必须删除并重新索引文档。
          </span>
        </label>
        <div className="flex items-start gap-3 border border-border bg-surface-alt px-4 py-3">
          <Database className="mt-0.5 h-4 w-4 text-text-muted" />
          <p className="text-xs leading-5 text-text-muted">
            重排序不可用时系统会保留基础混合检索，但不会再把该结果显示成已经完成重排序。
          </p>
        </div>
        <NumberField
          id="retrieval-topk"
          label="向量检索 Top-K"
          value={state.retrievalTopK}
          min={1}
          max={100}
          onChange={state.setRetrievalTopK}
        />
        <NumberField
          id="rerank-topk"
          label="重排序 Top-K"
          value={state.rerankTopK}
          min={1}
          max={50}
          onChange={state.setRerankTopK}
        />
        <NumberField
          id="retrieval-min-score"
          label="语义相关性阈值"
          value={state.retrievalMinScore}
          min={0}
          max={1}
          step={0.05}
          onChange={state.setRetrievalMinScore}
        />
        <NumberField
          id="keyword-min-coverage"
          label="关键词覆盖阈值"
          value={state.keywordMinCoverage}
          min={0}
          max={1}
          step={0.05}
          onChange={state.setKeywordMinCoverage}
        />
      </div>
    </section>
  );
}

function StatusItem({
  icon: Icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: typeof ScanSearch;
  label: string;
  value: string;
  detail: string;
  tone: "success" | "warning" | "muted";
}) {
  const valueClassName =
    tone === "success"
      ? "text-emerald-700 dark:text-emerald-300"
      : tone === "warning"
        ? "text-amber-700 dark:text-amber-300"
        : "text-text";
  return (
    <div className="flex items-start gap-3 bg-surface px-4 py-3">
      <Icon className="mt-0.5 h-4 w-4 text-text-muted" />
      <div className="min-w-0">
        <p className="text-xs text-text-muted">{label}</p>
        <p className={`mt-1 text-sm font-semibold ${valueClassName}`}>
          {value}
        </p>
        <p className="mt-1 text-xs leading-5 text-text-muted">{detail}</p>
      </div>
    </div>
  );
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label htmlFor={id} className="block">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        className={inputClassName}
      />
    </label>
  );
}
