import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  ObservabilityStatus,
  TraceDetailResponse,
  TraceListResponse,
  TraceStatus,
} from "@/types/observability";

interface TraceFilters {
  search: string;
  status: TraceStatus | "";
}

export function useObservabilityStatus() {
  return useQuery({
    queryKey: ["observability", "status"],
    queryFn: () => api.get<ObservabilityStatus>("/observability/status"),
    staleTime: 60_000,
  });
}

export function useTraceList(filters: TraceFilters) {
  const query = new URLSearchParams({
    limit: "100",
    offset: "0",
  });
  if (filters.search.trim()) query.set("search", filters.search.trim());
  if (filters.status) query.set("status", filters.status);

  return useQuery({
    queryKey: ["observability", "traces", filters],
    queryFn: () =>
      api.get<TraceListResponse>(`/observability/traces?${query.toString()}`),
    refetchInterval: (result) => (result.state.error ? 60_000 : 15_000),
  });
}

export function useTraceDetail(traceId: string | null) {
  return useQuery({
    queryKey: ["observability", "trace", traceId],
    queryFn: () =>
      api.get<TraceDetailResponse>(
        `/observability/traces/${encodeURIComponent(traceId ?? "")}`,
      ),
    enabled: Boolean(traceId),
  });
}

export function useDeleteTrace() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (traceId: string) =>
      api.delete<void>(
        `/observability/traces/${encodeURIComponent(traceId)}`,
      ),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["observability", "traces"],
      });
    },
  });
}

export function useClearTraces() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.delete<{ deleted: number }>("/observability/traces"),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["observability", "traces"],
      });
      queryClient.removeQueries({
        queryKey: ["observability", "trace"],
      });
    },
  });
}
