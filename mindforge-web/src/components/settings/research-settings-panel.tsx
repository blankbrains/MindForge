import { Check, Gauge, ShieldAlert } from "lucide-react";
import {
  useSettingsStore,
  type ResearchMode,
  type SourcePolicy,
} from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

const modes: Array<{
  id: ResearchMode;
  label: string;
  detail: string;
}> = [
  { id: "fast", label: "快速", detail: "单任务，不执行质量精炼" },
  { id: "balanced", label: "均衡", detail: "简单问题快速回答，复杂问题自动规划" },
  { id: "deep", label: "深度", detail: "完整 DAG、评审与精炼流程" },
];

export function ResearchSettingsPanel() {
  const state = useSettingsStore();

  return (
    <section
      className="space-y-6 border border-border bg-surface p-5 sm:p-6"
      role="tabpanel"
      aria-label="研究流程配置"
    >
      <div>
        <div className="flex items-center gap-2">
          <Gauge className="h-4 w-4 text-text-muted" />
          <h2 className="font-semibold">研究模式</h2>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
          {modes.map((mode) => {
            const selected = state.researchMode === mode.id;
            return (
              <button
                key={mode.id}
                type="button"
                aria-pressed={selected}
                onClick={() => state.setResearchMode(mode.id)}
                className={
                  "relative min-h-20 border px-3 py-3 text-left transition "
                  + (
                    selected
                      ? "border-primary-dark bg-primary/15 shadow-sm ring-2 ring-primary/30 dark:border-primary-light dark:bg-primary/25"
                      : "border-border bg-surface-alt hover:border-primary/60 hover:bg-primary/5"
                  )
                }
              >
                {selected && (
                  <Check
                    className="absolute right-3 top-3 h-4 w-4 text-primary-dark dark:text-primary-light"
                    aria-hidden="true"
                  />
                )}
                <span
                  className={
                    "block pr-6 text-sm font-semibold "
                    + (
                      selected
                        ? "text-primary-dark dark:text-primary-light"
                        : ""
                    )
                  }
                >
                  {mode.label}
                </span>
                <span
                  className={
                    "mt-1 block text-xs leading-5 "
                    + (selected ? "text-text" : "text-text-muted")
                  }
                >
                  {mode.detail}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium">研究来源</span>
          <select
            value={state.sourcePolicy}
            onChange={(event) =>
              state.setSourcePolicy(event.target.value as SourcePolicy)
            }
            className={inputClassName}
          >
            <option value="auto">自动选择知识库与联网搜索</option>
            <option value="knowledge_base">仅知识库</option>
            <option value="web">仅联网搜索</option>
          </select>
        </label>
        <ToggleRow
          label="研究失败后检索回退"
          detail="回退结果会明确标记为降级，不再作为正常回答评分"
          checked={state.fallbackEnabled}
          onChange={state.setFallbackEnabled}
        />
      </div>

      <div className="border-t border-border pt-5">
        <h3 className="text-sm font-semibold">执行预算</h3>
        <div className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
          <NumberField
            id="agent-max-subtasks"
            label="最大子任务数"
            value={state.maxSubtasks}
            min={1}
            max={20}
            onChange={state.setMaxSubtasks}
          />
          <NumberField
            id="agent-max-tool-calls"
            label="工具调用总预算"
            value={state.maxToolCallsTotal}
            min={1}
            max={100}
            onChange={state.setMaxToolCallsTotal}
          />
          <NumberField
            id="agent-queue-timeout"
            label="工具排队超时（秒）"
            detail="等待研究或工具并发槽位的最长时间"
            value={state.queueTimeout}
            min={1}
            max={600}
            onChange={state.setQueueTimeout}
          />
          <NumberField
            id="agent-max-iter"
            label="Researcher 最大轮次"
            value={state.maxIterations}
            min={1}
            max={20}
            onChange={state.setMaxIterations}
          />
          <NumberField
            id="agent-refine-rounds"
            label="最大精炼轮次"
            value={state.maxRefineRounds}
            min={0}
            max={5}
            onChange={state.setMaxRefineRounds}
          />
          <NumberField
            id="agent-subtask-timeout"
            label="子任务超时（秒）"
            value={state.subtaskTimeout}
            min={10}
            max={600}
            onChange={state.setSubtaskTimeout}
          />
          <NumberField
            id="agent-llm-timeout"
            label="单次模型调用超时（秒）"
            value={state.llmRequestTimeout}
            min={5}
            max={600}
            onChange={state.setLLMRequestTimeout}
          />
          <NumberField
            id="sandbox-timeout"
            label="代码执行超时（秒）"
            detail="单次 code_executor 沙箱运行的最长时间"
            value={state.sandboxTimeout}
            min={5}
            max={60}
            onChange={state.setSandboxTimeout}
          />
          <NumberField
            id="agent-research-timeout"
            label="研究总超时（秒）"
            value={state.researchTimeout}
            min={30}
            max={3600}
            onChange={state.setResearchTimeout}
          />
          <NumberField
            id="agent-threshold"
            label="Critic 评判阈值"
            value={state.criticThreshold}
            min={0}
            max={10}
            step={0.1}
            onChange={state.setCriticThreshold}
          />
        </div>
      </div>

      <div className="flex items-start gap-3 border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
        <p>
          “快速”和“均衡”模式会主动减少不必要的 Planner、Critic
          与精炼调用；均衡模式的单任务只评审、不自动重写。深度模式优先完整性，
          耗时和费用会更高。最大精炼轮次设为 0 时仍会进行质量评审，只是不触发
          报告重写。
        </p>
      </div>
    </section>
  );
}

function ToggleRow({
  label,
  detail,
  checked,
  onChange,
}: {
  label: string;
  detail: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex min-h-20 cursor-pointer items-center justify-between gap-4 border border-border bg-surface-alt px-4 py-3">
      <span>
        <span className="block text-sm font-medium">{label}</span>
        <span className="mt-1 block text-xs leading-5 text-text-muted">
          {detail}
        </span>
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-primary"
      />
    </label>
  );
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  step,
  detail,
  onChange,
}: {
  id: string;
  label: string;
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
      <label
        htmlFor={id}
        className="mb-1.5 block text-sm font-medium"
      >
        {label}
      </label>
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
