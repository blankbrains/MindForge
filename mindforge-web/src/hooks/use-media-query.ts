import { useState, useEffect } from "react";

type Breakpoint = "sm" | "md" | "lg" | "xl";

const queries: Record<Breakpoint, string> = {
  sm: "(min-width: 640px)",
  md: "(min-width: 768px)",
  lg: "(min-width: 1024px)",
  xl: "(min-width: 1280px)",
};

export function useMediaQuery(bp: Breakpoint): boolean {
  const [matches, setMatches] = useState(
    () => window.matchMedia(queries[bp]).matches,
  );

  useEffect(() => {
    const mql = window.matchMedia(queries[bp]);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [bp]);

  return matches;
}

export function useIsMobile(): boolean {
  return !useMediaQuery("md");
}
