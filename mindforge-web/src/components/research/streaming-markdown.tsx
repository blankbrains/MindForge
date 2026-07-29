import { memo, useCallback, useSyncExternalStore } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { markdownHighlightOptions } from "@/lib/markdown-highlight";
import { useResearchStore } from "@/store/research-store";

const configuredInterval = Number.parseInt(
  import.meta.env.VITE_STREAM_MARKDOWN_INTERVAL_MS || "",
  10,
);
const STREAM_MARKDOWN_INTERVAL_MS =
  Number.isFinite(configuredInterval) && configuredInterval >= 100
    ? configuredInterval
    : 350;

function getStreamingAnswerSnapshot(): string {
  return useResearchStore.getState().streamingAnswer;
}

function getServerStreamingAnswerSnapshot(): string {
  return "";
}

function useThrottledStreamingAnswer(): string {
  const subscribe = useCallback((notify: () => void) => {
    let timeout: ReturnType<typeof setTimeout> | null = null;
    const unsubscribe = useResearchStore.subscribe((state, previous) => {
      if (state.streamingAnswer === previous.streamingAnswer) return;
      if (state.streamingAnswer.length < previous.streamingAnswer.length) {
        if (timeout) clearTimeout(timeout);
        timeout = null;
        notify();
        return;
      }
      if (!timeout) {
        timeout = setTimeout(() => {
          timeout = null;
          notify();
        }, STREAM_MARKDOWN_INTERVAL_MS);
      }
    });
    return () => {
      if (timeout) clearTimeout(timeout);
      unsubscribe();
    };
  }, []);

  return useSyncExternalStore(
    subscribe,
    getStreamingAnswerSnapshot,
    getServerStreamingAnswerSnapshot,
  );
}

export const StreamingMarkdown = memo(function StreamingMarkdown({
  content,
}: {
  content: string;
}) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, markdownHighlightOptions]]}
        components={{
          img: ({ alt }) => (
            <span className="text-sm text-text-muted">
              [已阻止自动加载图片：{alt || "无说明"}]
            </span>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

export function StreamingAnswerPanel() {
  const content = useThrottledStreamingAnswer();
  if (!content) return null;

  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <StreamingMarkdown content={content} />
    </div>
  );
}
