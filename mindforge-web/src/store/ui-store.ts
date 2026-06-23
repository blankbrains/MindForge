import { create } from "zustand";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  // 1. localStorage 中用户手动选择的值优先
  try {
    const stored = localStorage.getItem("mindforge-theme");
    if (stored === "dark" || stored === "light") return stored;
  } catch { /* localStorage unavailable */ }
  // 2. 系统偏好
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }
  return "light";
}

function syncDomTheme(theme: Theme) {
  if (typeof document !== "undefined") {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }
}

// 初始化时立即同步 DOM 一次，避免首帧闪烁
const initialTheme = getInitialTheme();
syncDomTheme(initialTheme);

interface UIState {
  sidebarOpen: boolean;
  theme: Theme;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: Theme) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  theme: initialTheme,

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setTheme: (theme) => {
    syncDomTheme(theme);
    try { localStorage.setItem("mindforge-theme", theme); } catch { /* ignore */ }
    set({ theme });
  },
}));
