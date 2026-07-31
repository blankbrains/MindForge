import { type FormEvent, type MouseEvent } from "react";
import { Send, Square } from "lucide-react";

interface QueryInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (task: string) => void;
  isRunning: boolean;
  onCancel: () => void;
  retrievalOnly?: boolean;
}

export function QueryInput({
  value,
  onChange,
  onSubmit,
  isRunning,
  onCancel,
  retrievalOnly = false,
}: QueryInputProps) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim() && !isRunning) {
      onSubmit(value.trim());
    }
  };

  const handleCancel = (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    onCancel();
  };

  const buttonClassName =
    "inline-flex min-w-28 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative rounded-xl border border-border bg-surface shadow-sm transition-shadow focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary/50">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="输入你的研究问题… 例如：量子计算在药物研发中的应用前景如何？"
          rows={3}
          className="w-full resize-none rounded-xl border-0 bg-transparent px-5 py-4 text-base text-text placeholder:text-text-muted/60 focus:ring-0 focus:outline-none"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              e.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <div className="flex items-center justify-between border-t border-border px-4 py-3">
          <span className="text-xs text-text-muted">
            按 Enter 提交 · Shift+Enter 换行
          </span>
          {isRunning ? (
            <button
              key="cancel"
              type="button"
              onClick={handleCancel}
              className={`${buttonClassName} bg-red-500 hover:bg-red-600`}
            >
              <Square className="h-4 w-4 fill-current" />
              停止研究
            </button>
          ) : (
            <button
              key="submit"
              type="submit"
              disabled={!value.trim()}
              className={`${buttonClassName} bg-primary hover:bg-primary-dark`}
            >
              <>
                <Send className="h-4 w-4" />
                {retrievalOnly ? "知识库检索" : "开始研究"}
              </>
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
