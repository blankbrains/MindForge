import {
  memo,
  useCallback,
  useMemo,
  useSyncExternalStore,
} from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import type { Root } from "mdast";
import type { Plugin } from "unified";
import {
  normalizeCitationSources,
  safeHttpUrl,
} from "@/lib/citations";
import { markdownHighlightOptions } from "@/lib/markdown-highlight";
import { useResearchStore } from "@/store/research-store";
import type { CitationSource } from "@/types/research";

const configuredInterval = Number.parseInt(
  import.meta.env.VITE_STREAM_MARKDOWN_INTERVAL_MS || "",
  10,
);
const STREAM_MARKDOWN_INTERVAL_MS =
  Number.isFinite(configuredInterval) && configuredInterval >= 100
    ? configuredInterval
    : 350;

interface MarkdownNode {
  type: string;
  value?: string;
  url?: string;
  title?: string | null;
  data?: {
    hProperties?: Record<string, unknown>;
  };
  children?: MarkdownNode[];
}

interface CitationPluginOptions {
  targets: Record<string, string>;
}

const CITATION_PATTERN = /\[(\d+)\]/g;
const CITATION_SKIP_NODES = new Set([
  "code",
  "inlineCode",
  "link",
  "linkReference",
]);

function citationTextNodes(
  value: string,
  targets: Record<string, string>,
): MarkdownNode[] {
  const nodes: MarkdownNode[] = [];
  let cursor = 0;
  CITATION_PATTERN.lastIndex = 0;

  for (const match of value.matchAll(CITATION_PATTERN)) {
    const marker = match[0];
    const index = match[1];
    const start = match.index;
    const target = targets[index];
    if (!target) continue;
    if (start > cursor) {
      nodes.push({ type: "text", value: value.slice(cursor, start) });
    }
    nodes.push({
      type: "link",
      url: target,
      title: null,
      data: {
        hProperties: {
          className: "citation-link",
          "aria-label": `查看来源 ${index}`,
        },
      },
      children: [{ type: "text", value: marker }],
    });
    cursor = start + marker.length;
  }

  if (cursor === 0) return [{ type: "text", value }];
  if (cursor < value.length) {
    nodes.push({ type: "text", value: value.slice(cursor) });
  }
  return nodes;
}

function transformCitationNodes(
  node: MarkdownNode,
  targets: Record<string, string>,
): void {
  if (CITATION_SKIP_NODES.has(node.type) || !node.children) return;

  const transformed: MarkdownNode[] = [];
  for (const child of node.children) {
    if (child.type === "text" && typeof child.value === "string") {
      transformed.push(...citationTextNodes(child.value, targets));
      continue;
    }
    transformCitationNodes(child, targets);
    transformed.push(child);
  }
  node.children = transformed;
}

const remarkCitationLinks: Plugin<[CitationPluginOptions], Root> = (
  options,
) => {
  return (tree) => {
    transformCitationNodes(
      tree as unknown as MarkdownNode,
      options.targets,
    );
  };
};

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
  sources,
}: {
  content: string;
  sources?: CitationSource[];
}) {
  const normalizedSources = useMemo(
    () => normalizeCitationSources(sources),
    [sources],
  );
  const citationTargets = useMemo(
    () => Object.fromEntries(
      normalizedSources.map((source) => [
        String(source.index),
        source.url || `#research-source-${source.index}`,
      ]),
    ),
    [normalizedSources],
  );

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[
          remarkGfm,
          [remarkCitationLinks, { targets: citationTargets }],
        ]}
        rehypePlugins={[[rehypeHighlight, markdownHighlightOptions]]}
        components={{
          table: ({ children }) => (
            <div className="markdown-table-scroll">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href, node: _node, ...props }) => {
            const externalUrl = safeHttpUrl(href);
            return (
              <a
                {...props}
                href={externalUrl || href}
                target={externalUrl ? "_blank" : undefined}
                rel={externalUrl ? "noreferrer noopener" : undefined}
              >
                {children}
              </a>
            );
          },
          img: ({ alt }) => (
            <span className="text-sm text-text-muted">
              [已阻止自动加载图片：{alt || "无说明"}]
            </span>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
      {normalizedSources.length > 0 && (
        <section
          className="research-sources"
          aria-labelledby="research-sources-title"
        >
          <h2 id="research-sources-title">来源</h2>
          <ol>
            {normalizedSources.map((source) => (
              <li
                id={`research-source-${source.index}`}
                key={source.index}
              >
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    {source.title}
                  </a>
                ) : (
                  <span>{source.title}</span>
                )}
                {source.source && source.source !== source.title && (
                  <span className="research-source-kind">
                    {source.source}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
});

export function StreamingAnswerPanel() {
  const content = useThrottledStreamingAnswer();
  if (!content) return null;

  return (
    <div className="rounded-lg border border-border bg-surface p-5 sm:p-6">
      <StreamingMarkdown content={content} />
    </div>
  );
}
