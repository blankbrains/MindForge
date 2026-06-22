import { create } from "zustand";

type Theme = "light" | "dark";

interface UIState {
  sidebarOpen: boolean;
  theme: Theme;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: Theme) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  theme: (typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-color-scheme: dark)").matches)
    ? "dark"
    : "light",

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setTheme: (theme) => {
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
    set({ theme });
  },
}));
