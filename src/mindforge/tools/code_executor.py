"""Sandboxed Python code execution tool."""

from __future__ import annotations

import io
import signal
import textwrap
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

from mindforge.tools.base import BaseTool, ToolResult


# Forbidden keywords / imports that could be dangerous in a sandbox
FORBIDDEN_KEYWORDS: list[str] = [
    "__import__",
    "__builtins__",
    "eval",
    "exec",
    "compile",
    "open",
    "file",
    "breakpoint",
]

FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "ptty",
    "fcntl",
)

FORBIDDEN_MODULES: tuple[str, ...] = (
    "subprocess",
    "multiprocessing",
    "socket",
    "ctypes",
    "signal",
    "ptty",
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    # NOTE: getattr, type, object, super, issubclass, __import__ are EXCLUDED
    # — they enable Python object-model sandbox escapes.
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    # "issubclass": issubclass,  ← REMOVED — enables class hierarchy traversal
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    # "object": object,  ← REMOVED — enables __class__ chain sandbox escape
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    # "super": super,  ← REMOVED — enables parent-class access for sandbox escape
    "tuple": tuple,
    # "type": type,  ← REMOVED — enables dynamic class creation for sandbox escape
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ImportError": ImportError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
}


class SandboxTimeout(Exception):
    """Raised when code execution exceeds the allowed timeout."""


class SandboxViolation(Exception):
    """Raised when code contains forbidden constructs."""


class CodeExecutor(BaseTool):
    """Executes Python code in a restricted sandbox.

    Captures stdout and stderr. Enforces a timeout and blocks dangerous
    operations such as subprocess, socket, and file I/O.
    """

    name = "code_executor"
    description = (
        "Execute Python code in a restricted sandbox. Returns stdout and stderr. "
        "The sandbox blocks file I/O, subprocesses, sockets, and other system-level "
        "operations. Use this for calculations, data processing, and algorithm "
        "prototyping only."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds.",
                "default": 30,
                "minimum": 1,
                "maximum": 120,
            },
            "vars": {
                "type": "object",
                "description": "Dictionary of variables to inject into the sandbox globals.",
                "default": {},
            },
        },
        "required": ["code"],
    }

    def __init__(self, forbidden_keywords: Optional[list[str]] = None) -> None:
        super().__init__()
        self._forbidden_keywords = forbidden_keywords or FORBIDDEN_KEYWORDS

    def execute(
        self,
        code: str,
        timeout: int = 30,
        vars: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()

        if not code or not code.strip():
            return ToolResult(
                success=False,
                error="Code must be a non-empty string.",
            )

        # --- Pre-execution checks ---
        violation = self._check_forbidden(code)
        if violation:
            return ToolResult(
                success=False,
                error=f"Sandbox violation: {violation}",
            )

        # --- Prepare sandbox globals ---
        sandbox_globals: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__sandbox__",
        }
        if vars:
            # Only inject safe, serializable types
            for k, v in vars.items():
                if isinstance(k, str) and k.isidentifier():
                    sandbox_globals[k] = v

        # --- Execute ---
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        compiled = self._compile_code(code)
        if isinstance(compiled, ToolResult):
            return compiled  # Compilation error

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                self._run_with_timeout(compiled, sandbox_globals, timeout)
        except SandboxTimeout:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout}s.",
                execution_time_ms=elapsed,
            )
        except SandboxViolation as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Sandbox violation during execution: {exc}",
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            stderr_capture.write(f"{type(exc).__name__}: {exc}\n")
            stderr_capture.write(traceback.format_exc())
            return ToolResult(
                success=False,
                error=str(exc),
                output=stdout_capture.getvalue(),
                data={"stderr": stderr_capture.getvalue()},
                execution_time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Check for truncated output
        truncated = False
        if len(stdout) > 100_000:
            stdout = stdout[:100_000] + "\n... [stdout truncated at 100KB]"
            truncated = True

        return ToolResult(
            success=True,
            output=stdout,
            data={
                "stderr": stderr,
                "return_value": sandbox_globals.get("_return", None),
                "sandbox_vars": {
                    k: v
                    for k, v in sandbox_globals.items()
                    if not k.startswith("_") and k != "None"
                },
            },
            truncated=truncated,
            execution_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Compiled patterns for word-boundary matching (avoids false positives)
    _FORBIDDEN_PATTERNS: list[Any] = None  # populated lazily

    def _check_forbidden(self, code: str) -> Optional[str]:
        """Check code for forbidden keywords using word-boundary matching.

        Uses ``\b`` regex anchors so that e.g. "eval" matches only the
        standalone identifier, not "evaluate" or "eval_expression".
        """
        import re

        # Lazy compilation of word-boundary patterns
        if self._FORBIDDEN_PATTERNS is None:
            self._FORBIDDEN_PATTERNS = []
            for kw in FORBIDDEN_KEYWORDS:
                self._FORBIDDEN_PATTERNS.append(
                    (re.compile(r'\b' + re.escape(kw) + r'\b'), f"Forbidden keyword: {kw}")
                )
            for imp in FORBIDDEN_IMPORTS:
                self._FORBIDDEN_PATTERNS.append(
                    (re.compile(re.escape(imp)), f"Forbidden attribute access: {imp}")
                )
            for mod in FORBIDDEN_MODULES:
                self._FORBIDDEN_PATTERNS.append(
                    (re.compile(r'\bimport\s+' + re.escape(mod) + r'\b|from\s+' + re.escape(mod) + r'\b'),
                     f"Forbidden module: {mod}")
                )

        lower_code = code.lower()
        for pattern, msg in self._FORBIDDEN_PATTERNS:
            if pattern.search(lower_code):
                return msg

        return None

    def _compile_code(self, code: str) -> Any:
        """Compile code; return ToolResult on error."""
        try:
            return compile(
                textwrap.dedent(code),
                filename="<sandbox>",
                mode="exec",
            )
        except SyntaxError as exc:
            return ToolResult(
                success=False,
                error=f"Syntax error: {exc}",
                data={
                    "lineno": exc.lineno,
                    "offset": exc.offset,
                    "text": exc.text,
                },
            )

    def _run_with_timeout(
        self,
        compiled: Any,
        sandbox_globals: dict[str, Any],
        timeout: int,
    ) -> None:
        """Run compiled code with a timeout using signal.SIGALRM.

        On Windows (where SIGALRM is unavailable), falls back to a
        threading-based timeout.
        """
        if hasattr(signal, "SIGALRM"):
            self._run_with_sigalrm(compiled, sandbox_globals, timeout)
        else:
            self._run_with_thread_timeout(compiled, sandbox_globals, timeout)

    def _run_with_sigalrm(
        self,
        compiled: Any,
        sandbox_globals: dict[str, Any],
        timeout: int,
    ) -> None:
        """Unix: use SIGALRM for timeout."""

        def handler(signum: int, frame: Any) -> None:
            raise SandboxTimeout()

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            exec(compiled, sandbox_globals)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _run_with_thread_timeout(
        self,
        compiled: Any,
        sandbox_globals: dict[str, Any],
        timeout: int,
    ) -> None:
        """Windows fallback: use threading.Timer to enforce timeout.

        Note: This approach cannot forcibly kill a stuck thread, but
        if the code completes within the window it works correctly.
        For true isolation, spawn a subprocess.
        """
        import threading

        result: list[Any] = []
        exception: list[Optional[BaseException]] = [None]
        finished = threading.Event()

        def target() -> None:
            try:
                exec(compiled, sandbox_globals)
                result.append(True)
            except BaseException as exc:
                exception[0] = exc
            finally:
                finished.set()

        t = threading.Thread(target=target, daemon=True)
        t.start()

        if not finished.wait(timeout=timeout):
            raise SandboxTimeout()

        if exception[0] is not None:
            raise exception[0]  # type: ignore[operator]
