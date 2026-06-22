import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { StatsResponse } from "@/types/api";

export function useStats() {
  return useQuery<StatsResponse>({
    queryKey: ["stats"],
    queryFn: () => api.get("/stats"),
    refetchInterval: 30_000,
  });
}
