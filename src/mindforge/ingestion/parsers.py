"""多格式文档解析器 — 支持 PDF/DOCX/HTML/MD/TXT"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import hashlib
import logging
import re

logger = logging.getLogger(__name__)


def _read_text_with_fallback(path: Path) -> str:
    """读取文本文件，依次尝试 utf-8 / gbk / latin-1 编码。"""
    encodings = ["utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # 最后兜底
    return path.read_text(encoding="utf-8", errors="replace")


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

    def __init__(self) -> None:
        from mindforge.config import get_settings

        self._limits = get_settings().api

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}，支持: {self.SUPPORTED_EXTENSIONS}")
        st = path.stat()
        if (
            suffix in {".html", ".htm", ".md", ".txt"}
            and st.st_size
            > self._limits.max_text_file_mb * 1024 * 1024
        ):
            raise ValueError(
                "Text document exceeds the configured parser size limit."
            )
        parser = self._get_parser(suffix)
        content, sections, metadata = parser(path)
        if len(content) > self._limits.max_parsed_chars:
            raise ValueError(
                "Parsed document exceeds the configured character limit."
            )
        doc_id = hashlib.md5(
            f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{content[:256]}".encode()
        ).hexdigest()[:12]
        metadata.update({"source": path.name, "file_type": suffix, "size_bytes": st.st_size})
        logger.info(f"已解析: {path.name} ({len(content)} 字符)")
        return ParsedDocument(doc_id=doc_id, filename=path.name, content=content, sections=sections, metadata=metadata)

    def _get_parser(self, suffix: str):
        parsers = {".pdf": self._parse_pdf, ".docx": self._parse_docx, ".html": self._parse_html,
                   ".htm": self._parse_html, ".md": self._parse_markdown, ".txt": self._parse_text}
        return parsers[suffix]

    def _parse_pdf(self, path: Path):
        import pdfplumber
        import warnings
        # pdfminer 对某些嵌入字体会输出 FontBBox 警告，不影响内容提取
        warnings.filterwarnings("ignore", message=".*FontBBox.*")
        warnings.filterwarnings("ignore", message=".*font descriptor.*")
        content_parts, sections = [], []

        with pdfplumber.open(str(path)) as pdf:
            total = len(pdf.pages)
            if total > self._limits.max_pdf_pages:
                raise ValueError(
                    "PDF exceeds the configured page limit of "
                    f"{self._limits.max_pdf_pages}."
                )
            if total <= 10:
                # 小 PDF 直接串行
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    content_parts.append(text)
                    sections.append({"title": f"第 {i+1} 页", "content": text, "level": 0})
                    if (
                        sum(len(part) for part in content_parts)
                        > self._limits.max_parsed_chars
                    ):
                        raise ValueError(
                            "PDF text exceeds the configured character limit."
                        )
            else:
                # 大 PDF 并行解析 — 线程数取 min(8, CPU 核数)
                workers = min(8, os.cpu_count() or 4)
                logger.info("PDF 并行解析: %d 页, %d 线程", total, workers)

                def _extract_page(i: int) -> tuple[int, str]:
                    try:
                        return i, pdf.pages[i].extract_text() or ""
                    except Exception:
                        return i, ""

                total_chars = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    batch_size = workers * 2
                    for start in range(0, total, batch_size):
                        stop = min(total, start + batch_size)
                        results = list(
                            pool.map(_extract_page, range(start, stop))
                        )
                        results.sort(key=lambda item: item[0])
                        for i, text in results:
                            total_chars += len(text)
                            if total_chars > self._limits.max_parsed_chars:
                                raise ValueError(
                                    "PDF text exceeds the configured "
                                    "character limit."
                                )
                            content_parts.append(text)
                            sections.append({
                                "title": f"第 {i+1} 页",
                                "content": text,
                                "level": 0,
                            })
                logger.info(
                    "PDF 解析完成: %d 页, %d 字符",
                    total,
                    total_chars,
                )

        return "\n".join(content_parts), sections, {"pages": len(content_parts)}

    def _parse_docx(self, path: Path):
        from zipfile import BadZipFile, ZipFile

        try:
            with ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > self._limits.max_docx_parts:
                    raise ValueError(
                        "DOCX package exceeds the configured part limit."
                    )
                expanded_size = sum(info.file_size for info in infos)
                if (
                    expanded_size
                    > self._limits.max_docx_uncompressed_mb
                    * 1024
                    * 1024
                ):
                    raise ValueError(
                        "DOCX package exceeds the configured expanded-size "
                        "limit."
                    )
        except BadZipFile as exc:
            raise ValueError("Invalid DOCX package.") from exc

        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        content_parts, sections = [], []
        total_chars = 0
        for para in doc.paragraphs:
            if para.text.strip():
                total_chars += len(para.text)
                if total_chars > self._limits.max_parsed_chars:
                    raise ValueError(
                        "DOCX text exceeds the configured character limit."
                    )
                content_parts.append(para.text)
                if para.style.name.startswith("Heading"):
                    try:
                        level_str = para.style.name[len("Heading"):].strip()
                        level = int(level_str.split()[0]) if level_str else 1
                    except (ValueError, IndexError):
                        level = 1
                    sections.append({"title": para.text, "content": para.text, "level": level})
        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    total_chars += len(row_text)
                    if total_chars > self._limits.max_parsed_chars:
                        raise ValueError(
                            "DOCX text exceeds the configured character "
                            "limit."
                        )
                    content_parts.append(row_text)
        return "\n".join(content_parts), sections, {}

    def _parse_html(self, path: Path):
        from bs4 import BeautifulSoup
        raw = _read_text_with_fallback(path)
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text, [], {"title": soup.title.string if soup.title else ""}

    def _parse_markdown(self, path: Path):
        content = _read_text_with_fallback(path)
        sections = []
        for line in content.split("\n"):
            m = re.match(r'^(#{1,6})\s', line)
            if m:
                level = len(m.group(1))
                sections.append({"title": line.lstrip("#").strip(), "content": "", "level": level})
        return content, sections, {}

    def _parse_text(self, path: Path):
        return _read_text_with_fallback(path), [], {}


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
