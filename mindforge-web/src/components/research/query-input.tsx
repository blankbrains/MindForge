import { type FormEvent } from "react";
import { Send, Loader2 } from "lucide-react";

interface QueryInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (task: string) => void;
  disabled: boolean;
}

export function QueryInput({ value, onChange, onSubmit, disabled }: QueryInputProps) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim() && !disabled) {
      onSubmit(value.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative rounded-xl border border-border bg-surface shadow-sm transition-shadow focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary/50">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="输入你的研究问题… 例如：量子计算在药物研发中的应用前景如何？"
          rows={3}
          disabled={disabled}
          className="w-full resize-none rounded-xl border-0 bg-transparent px-5 py-4 text-base text-text placeholder:text-text-muted/60 focus:ring-0 focus:outline-none disabled:opacity-50"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <div className="flex items-center justify-between border-t border-border px-4 py-3">
          <span className="text-xs text-text-muted">
            按 Enter 提交 · Shift+Enter 换行
          </span>
          <button
            type="submit"
            disabled={!value.trim() || disabled}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:opacity-50"
          >
            {disabled ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                处理中…
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                开始研究
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
