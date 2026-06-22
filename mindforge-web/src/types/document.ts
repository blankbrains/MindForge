export interface DocumentItem {
  doc_id: string;
  filename: string;
  chunk_count: number;
  status: string;
  source?: string;
  size_bytes?: number;
  indexed_at?: string;
}

export interface IndexResponse {
  doc_id: string;
  filename: string;
  chunk_count: number;
  status: string;
}
