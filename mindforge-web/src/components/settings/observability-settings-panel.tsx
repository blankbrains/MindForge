import { Activity, EyeOff, Server } from "lucide-react";
import { useSettingsStore } from "@/store/settings-store";

const inputClassName =
  "w-full rounded-md border border-border bg-surface-alt px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

export function ObservabilitySettingsPanel() {
  const state = useSettingsStore();
  const langfuseConfigured = Boolean(
    state.langfuseHost
      && state.langfusePublicKey
      && state.langfuseSecretKey,
  );

  return (
    <section
      className="space-y-6 border border-border bg-surface p-5 sm:p-6"
      role="tabpanel"
      aria-label="可观测配置"
    >
      <div className="grid grid-cols-1 gap-px border border-border bg-border sm:grid-cols-2">
        <StatusItem
          icon={Activity}
          label="本地 Trace"
          value="已启用"
        />
        <StatusItem
          icon={Server}
          label="Langfuse"
          value={langfuseConfigured ? "凭证已配置" : "未配置"}
        />
      </div>

      <div>
        <h2 className="font-semibold">Langfuse 连接</h2>
        <p className="mt-1 text-xs leading-5 text-text-muted">
          页面保存与直接编辑服务端 .env 等效，密钥重新加载后只返回掩码。
        </p>
        <div className="mt-4 grid grid-cols-1 gap-4">
          <TextField
            id="langfuse-host"
            label="Host"
            value={state.langfuseHost}
            onChange={state.setLangfuseHost}
          />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <TextField
              id="langfuse-public-key"
              label="Public Key"
              value={state.langfusePublicKey}
              onChange={state.setLangfusePublicKey}
            />
            <TextField
              id="langfuse-secret-key"
              label="Secret Key"
              value={state.langfuseSecretKey}
              onChange={state.setLangfuseSecretKey}
              secret
            />
          </div>
        </div>
      </div>

      <div className="border-t border-border pt-5">
        <h3 className="text-sm font-semibold">本地数据</h3>
        <div className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium">
              Trace 保留
            </span>
            <select
              value={state.traceRetentionDays}
              onChange={(event) =>
                state.setTraceRetentionDays(Number(event.target.value))
              }
              className={inputClassName}
            >
              <option value={0}>永久保留</option>
              <option value={7}>7 天</option>
              <option value={30}>30 天</option>
              <option value={90}>90 天</option>
              <option value={365}>1 年</option>
            </select>
          </label>
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium">
              研究历史上限
            </span>
            <input
              type="number"
              min={0}
              max={100_000}
              value={state.maxHistoryEntries}
              onChange={(event) =>
                state.setMaxHistoryEntries(Number(event.target.value))
              }
              className={inputClassName}
            />
            <span className="mt-1 block text-xs text-text-muted">
              0 表示永久保留，仍可在历史页面手动删除。
            </span>
          </label>
        </div>
      </div>

      <label className="flex cursor-pointer items-start justify-between gap-4 border border-border bg-surface-alt px-4 py-3">
        <span className="flex gap-3">
          <EyeOff className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" />
          <span>
            <span className="block text-sm font-medium">采集研究正文</span>
            <span className="mt-1 block text-xs leading-5 text-text-muted">
              默认只记录结构、耗时、Token、费用和错误。启用后正文可能进入本地 Trace
              与 Langfuse。
            </span>
          </span>
        </span>
        <input
          type="checkbox"
          checked={state.observabilityCaptureContent}
          onChange={(event) =>
            state.setObservabilityCaptureContent(event.target.checked)
          }
          className="mt-1 h-4 w-4 accent-primary"
        />
      </label>
    </section>
  );
}

function StatusItem({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-3 bg-surface px-4 py-3">
      <Icon className="mt-0.5 h-4 w-4 text-text-muted" />
      <div>
        <p className="text-xs text-text-muted">{label}</p>
        <p className="mt-1 text-sm font-semibold">{value}</p>
      </div>
    </div>
  );
}

function TextField({
  id,
  label,
  value,
  onChange,
  secret = false,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  secret?: boolean;
}) {
  return (
    <label htmlFor={id} className="block">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      <input
        id={id}
        type={secret && !value.startsWith("***") ? "password" : "text"}
        value={value}
        autoComplete="off"
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
        className={inputClassName}
      />
    </label>
  );
}
