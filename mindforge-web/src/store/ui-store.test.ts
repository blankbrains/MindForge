import { afterEach, describe, expect, it } from "vitest";

import { useUIStore } from "@/store/ui-store";

describe("UI store", () => {
  afterEach(() => {
    useUIStore.getState().setSidebarOpen(false);
  });

  it("keeps the mobile sidebar closed until the user opens it", () => {
    expect(useUIStore.getState().sidebarOpen).toBe(false);

    useUIStore.getState().toggleSidebar();

    expect(useUIStore.getState().sidebarOpen).toBe(true);
  });
});
