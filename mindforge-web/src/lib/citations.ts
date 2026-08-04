import type { CitationSource } from "@/types/research";

const MAX_SOURCES = 200;
const GENERIC_SOURCE_TITLES = new Set([
  "",
  "untitled",
  "web",
  "web source",
]);

function boundedString(value: unknown, maxLength: number): string {
  return typeof value === "string"
    ? value.trim().slice(0, maxLength)
    : "";
}

export function safeHttpUrl(value: unknown): string {
  const candidate = boundedString(value, 4_096);
  if (!candidate) return "";
  try {
    const url = new URL(candidate);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.href
      : "";
  } catch {
    return "";
  }
}

function isGenericSourceTitle(value: string): boolean {
  return GENERIC_SOURCE_TITLES.has(value.trim().toLocaleLowerCase());
}

function titleFromUrl(value: string, index: number): string {
  if (!value) return `来源 ${index}`;
  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./i, "");
    const pathParts = url.pathname
      .split("/")
      .map((part) => decodeURIComponent(part).trim())
      .filter(Boolean);
    const rawSlug = pathParts.at(-1) ?? "";
    const slug = rawSlug.replace(/\.(?:html?|php|aspx?)$/i, "");
    if (
      !slug
      || /^(?:article|detail|index|page)$/i.test(slug)
      || /^\d+$/.test(slug)
    ) {
      return host || `来源 ${index}`;
    }
    const readable = slug.replace(/[-_]+/g, " ").replace(/\s+/g, " ").trim();
    return readable ? `${host} - ${readable}` : host;
  } catch {
    return `来源 ${index}`;
  }
}

export function normalizeCitationSources(value: unknown): CitationSource[] {
  if (!Array.isArray(value)) return [];

  const sources: CitationSource[] = [];
  const seenIndices = new Set<number>();
  for (const item of value) {
    if (sources.length >= MAX_SOURCES) break;
    if (!item || typeof item !== "object") continue;

    const record = item as Record<string, unknown>;
    const index = Number(record.index);
    if (
      !Number.isInteger(index)
      || index < 1
      || index > 100_000
      || seenIndices.has(index)
    ) {
      continue;
    }
    seenIndices.add(index);

    const chunkId = boundedString(record.chunk_id, 512);
    const docId = boundedString(
      record.doc_id ?? record.document_id,
      512,
    );
    const url = safeHttpUrl(record.url);
    const rawTitle = boundedString(record.title, 1_000);
    const rawSource = boundedString(record.source, 200);
    sources.push({
      index,
      title:
        (!isGenericSourceTitle(rawTitle) ? rawTitle : "")
        || (!isGenericSourceTitle(rawSource) ? rawSource : "")
        || titleFromUrl(url, index),
      url,
      source: rawSource,
      ...(chunkId ? { chunk_id: chunkId } : {}),
      ...(docId ? { doc_id: docId } : {}),
    });
  }
  return sources.sort((left, right) => left.index - right.index);
}
