import dockerfile from "highlight.js/lib/languages/dockerfile";
import powershell from "highlight.js/lib/languages/powershell";
import ini from "highlight.js/lib/languages/ini";
import { common } from "lowlight";
import type { Options } from "rehype-highlight";

const AUTO_DETECT_LANGUAGES = [
  "bash",
  "c",
  "cpp",
  "csharp",
  "css",
  "diff",
  "dockerfile",
  "go",
  "graphql",
  "ini",
  "java",
  "javascript",
  "json",
  "kotlin",
  "lua",
  "makefile",
  "markdown",
  "objectivec",
  "perl",
  "php",
  "powershell",
  "python",
  "r",
  "ruby",
  "rust",
  "scss",
  "shell",
  "sql",
  "swift",
  "toml",
  "typescript",
  "xml",
  "yaml",
];

export const markdownHighlightOptions: Options = {
  aliases: {
    dockerfile: ["docker"],
    powershell: ["ps1"],
  },
  detect: true,
  languages: {
    ...common,
    dockerfile,
    powershell,
    toml: ini,
  },
  plainText: ["text", "txt", "plaintext"],
  subset: AUTO_DETECT_LANGUAGES,
};
