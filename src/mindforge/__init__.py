"""MindForge: 自适应研究助理系统"""

from __future__ import annotations
import io
import sys

# ── 全局 UTF-8 编码垫片（确保控制台/文件读写不乱码）──
if sys.stdout.encoding is None or sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

__version__ = "1.0.0"
