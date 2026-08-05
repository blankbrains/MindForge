import { Database, Globe2, ScanSearch } from "lucide-react";
import { HelpTooltip } from "@/components/shared/tooltip";
import {
  useSettingsStore,
  type EmbeddingProvider,
} from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

export function RetrievalSettingsPanel() {
  const state = useSettingsStore();
  const protocolLabels = {
    none: "未启用",
    openai_responses: "Responses Web Search",
    kimi_builtin: "Kimi 内置联网",
    glm_web_search: "GLM Web Search",
  } as const;
  const webSearchValue = state.nativeWebSearchSupported
    ? "模型原生联网可用"
    : state.tavilyConfigured
      ? "Tavily 辅助搜索可用"
      : state.duckDuckGoEnabled
        ? "DuckDuckGo 辅助搜索已启用"
        : "当前无联网后端";
  const webSearchDetail = state.nativeWebSearchSupported
    ? `当前协议：${protocolLabels[state.nativeWebSearchProtocol]}`
    : state.webSearchAvailable
      ? "模型无原生联网时使用已配置的辅助搜索"
      : state.modelOnlyFallbackEnabled
        ? "研究会保留模型回答，并明确标记为无可核验引用"
        : "需要启用模型原生联网或配置可选辅助搜索";

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
          value={webSearchValue}
          detail={webSearchDetail}
          tone={
            state.webSearchAvailable
              ? "success"
              : state.modelOnlyFallbackEnabled
                ? "warning"
                : "muted"
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <div className="block">
          <div className="mb-1.5 flex items-center gap-1.5">
            <label
              htmlFor="retrieval-embedding"
              className="text-sm font-medium"
            >
              Embedding
            </label>
            <HelpTooltip
              label="Embedding 说明"
              content="把文档和问题转换为向量，用于语义检索。切换模型后向量空间会变化，因此已有文档必须重新索引。"
            />
          </div>
          <select
            id="retrieval-embedding"
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
        </div>
        <div className="flex items-start gap-3 border border-border bg-surface-alt px-4 py-3">
          <Database className="mt-0.5 h-4 w-4 text-text-muted" />
          <p className="text-xs leading-5 text-text-muted">
            重排序不可用时系统会保留基础混合检索，但不会再把该结果显示成已经完成重排序。
          </p>
        </div>
        <NumberField
          id="retrieval-topk"
          label="向量检索 Top-K"
          help="向量检索阶段最多取回的候选知识块数量。过小可能漏掉证据，过大会增加重排序成本和噪声。"
          value={state.retrievalTopK}
          min={1}
          max={100}
          onChange={state.setRetrievalTopK}
        />
        <NumberField
          id="rerank-topk"
          label="重排序 Top-K"
          help="CrossEncoder 精排后保留给 Agent 的知识块数量，不能大于向量检索候选规模。"
          value={state.rerankTopK}
          min={1}
          max={50}
          onChange={state.setRerankTopK}
        />
        <NumberField
          id="retrieval-min-score"
          label="语义相关性阈值"
          help="过滤语义相似度不足的向量结果。阈值越高越严格，可能提高精度但降低召回率。"
          value={state.retrievalMinScore}
          min={0}
          max={1}
          step={0.05}
          onChange={state.setRetrievalMinScore}
        />
        <NumberField
          id="keyword-min-coverage"
          label="关键词覆盖阈值"
          help="判断检索结果是否覆盖问题关键词的最低比例，用于识别相关性不足或证据缺失。"
          value={state.keywordMinCoverage}
          min={0}
          max={1}
          step={0.05}
          onChange={state.setKeywordMinCoverage}
        />
        <NumberField
          id="native-web-search-timeout"
          label="原生联网搜索超时（秒）"
          help="模型供应商原生联网能力的等待上限。超时后系统会按配置尝试辅助搜索或降级回答。"
          detail="模型供应商原生联网请求的最长等待时间"
          value={state.nativeWebSearchTimeoutSeconds}
          min={5}
          max={120}
          onChange={state.setNativeWebSearchTimeoutSeconds}
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
  help,
  value,
  min,
  max,
  step,
  detail,
  onChange,
}: {
  id: string;
  label: string;
  help?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  detail?: string;
  onChange: (value: number) => void;
}) {
  const descriptionId = detail ? `${id}-description` : undefined;
  return (
    <div className="block">
      <div className="mb-1.5 flex items-center gap-1.5">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        {help && <HelpTooltip content={help} label={`${label}说明`} />}
      </div>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-describedby={descriptionId}
        className={inputClassName}
      />
      {detail && (
        <span
          id={descriptionId}
          className="mt-1 block text-xs leading-5 text-text-muted"
        >
          {detail}
        </span>
      )}
    </div>
  );
}
