import type { CitationSource } from "@/types/research";

const MAX_SOURCES = 200;

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
    sources.push({
      index,
      title:
        boundedString(record.title, 1_000)
        || boundedString(record.source, 200)
        || `来源 ${index}`,
      url: safeHttpUrl(record.url),
      source: boundedString(record.source, 200),
      ...(chunkId ? { chunk_id: chunkId } : {}),
      ...(docId ? { doc_id: docId } : {}),
    });
  }
  return sources.sort((left, right) => left.index - right.index);
}
