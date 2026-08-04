import { describe, expect, it } from "vitest";
import { normalizeCitationSources } from "@/lib/citations";

describe("normalizeCitationSources", () => {
  it("derives a readable title for legacy generic web sources", () => {
    const [source] = normalizeCitationSources([
      {
        index: 1,
        title: "Web source",
        url: "https://www.infoworld.com/article/4071159/java-or-python-for-building-agents.html",
        source: "web",
      },
    ]);

    expect(source.title).toBe(
      "infoworld.com - java or python for building agents",
    );
  });

  it("preserves a concrete provider title", () => {
    const [source] = normalizeCitationSources([
      {
        index: 1,
        title: "Java or Python for building agents",
        url: "https://example.com/article",
        source: "web",
      },
    ]);

    expect(source.title).toBe("Java or Python for building agents");
  });
});
