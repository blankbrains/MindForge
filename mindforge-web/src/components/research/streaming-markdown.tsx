import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

export function StreamingMarkdown({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
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
}
