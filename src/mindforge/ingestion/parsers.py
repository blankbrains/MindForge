"""多格式文档解析器 — 支持 PDF/DOCX/HTML/MD/TXT"""
from __future__ import annotations
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    content: str
    sections: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    images: List[dict] = field(default_factory=list)


class DocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}，支持: {self.SUPPORTED_EXTENSIONS}")
        parser = self._get_parser(suffix)
        content, sections, metadata = parser(path)
        doc_id = hashlib.md5(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()[:12]
        metadata.update({"source": path.name, "file_type": suffix, "size_bytes": path.stat().st_size})
        logger.info(f"已解析: {path.name} ({len(content)} 字符)")
        return ParsedDocument(doc_id=doc_id, filename=path.name, content=content, sections=sections, metadata=metadata)

    def _get_parser(self, suffix: str):
        parsers = {".pdf": self._parse_pdf, ".docx": self._parse_docx, ".html": self._parse_html,
                   ".htm": self._parse_html, ".md": self._parse_markdown, ".txt": self._parse_text}
        return parsers[suffix]

    def _parse_pdf(self, path: Path):
        import pdfplumber
        content_parts, sections = [], []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                content_parts.append(text)
                sections.append({"title": f"第 {i+1} 页", "content": text, "level": 0})
        return "\n".join(content_parts), sections, {"pages": len(content_parts)}

    def _parse_docx(self, path: Path):
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        content_parts, sections = [], []
        for para in doc.paragraphs:
            if para.text.strip():
                content_parts.append(para.text)
                if para.style.name.startswith("Heading"):
                    try:
                        # Extract heading level number: "Heading 1" → 1, "Heading 2" → 2
                        level_str = para.style.name[len("Heading"):].strip()
                        level = int(level_str.split()[0]) if level_str else 1
                    except (ValueError, IndexError):
                        level = 1  # default for non-standard heading styles
                    sections.append({"title": para.text, "content": para.text, "level": level})
        return "\n".join(content_parts), sections, {}

    def _parse_html(self, path: Path):
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text, [], {"title": soup.title.string if soup.title else ""}

    def _parse_markdown(self, path: Path):
        content = path.read_text(encoding="utf-8")
        sections = []
        for line in content.split("\n"):
            if line.startswith("#"):
                level = len(line.split(" ")[0])
                sections.append({"title": line.lstrip("# "), "content": "", "level": level})
        return content, sections, {}

    def _parse_text(self, path: Path):
        return path.read_text(encoding="utf-8"), [], {}


class DirectoryParser:
    def __init__(self, parser: DocumentParser | None = None):
        self.parser = parser or DocumentParser()

    def parse_directory(self, dir_path: str | Path, recursive: bool = True) -> List[ParsedDocument]:
        docs = []
        base = Path(dir_path)
        pattern = "**/*" if recursive else "*"
        for fp in sorted(base.glob(pattern)):
            if fp.suffix.lower() in self.parser.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.parser.parse(fp)
                    docs.append(doc)
                except Exception as e:
                    logger.warning(f"解析失败 {fp.name}: {e}")
        logger.info(f"共解析 {len(docs)} 个文档")
        return docs
