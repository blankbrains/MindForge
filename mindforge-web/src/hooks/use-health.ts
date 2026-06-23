import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { HealthResponse } from "@/types/api";

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: () => api.get("/health"),
    refetchInterval: (query) => (query.state.error ? 60_000 : 15_000),
  });
}
