"""Sandboxed Python code execution tool."""

from __future__ import annotations

import ast
import json
import os
import site
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

from mindforge.tools.base import BaseTool, ToolResult
from mindforge.config import get_settings, resolve_project_path


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

FORBIDDEN_FILE_API_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "load",
        "dump",
        "save",
        "savez",
        "savez_compressed",
        "fromfile",
        "tofile",
        "memmap",
        "loadtxt",
        "genfromtxt",
        "savetxt",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_parquet",
        "read_pickle",
        "read_sas",
        "read_spss",
        "read_sql",
        "read_stata",
        "read_table",
        "read_xml",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_json",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
        "listdir",
        "scandir",
        "walk",
        "glob",
        "iglob",
        "iterdir",
        "rglob",
        "stat",
        "lstat",
        "readlink",
        "remove",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "mkdir",
        "makedirs",
        "removedirs",
        "chdir",
        "chmod",
        "chown",
        "truncate",
        "link",
        "symlink",
        "system",
        "popen",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "ctypeslib",
        "cdll",
        "windll",
        "oledll",
        "pydll",
        "CDLL",
        "PyDLL",
        "WinDLL",
        "OleDLL",
    }
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
                "default": 15,
                "minimum": 1,
                "maximum": 60,
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
        self._settings = get_settings()
        self._forbidden_keywords = forbidden_keywords or FORBIDDEN_KEYWORDS

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        vars: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()
        effective_timeout = (
            self._settings.sandbox.sandbox_timeout
            if timeout is None
            else timeout
        )

        if not code or not code.strip():
            return ToolResult(
                success=False,
                error="Code must be a non-empty string.",
            )
        if len(code) > self._settings.sandbox.max_code_length:
            return ToolResult(
                success=False,
                error=(
                    "Code exceeds the configured maximum length of "
                    f"{self._settings.sandbox.max_code_length} characters."
                ),
            )
        if (
            isinstance(effective_timeout, bool)
            or not isinstance(effective_timeout, int)
            or not 1
            <= effective_timeout
            <= self._settings.sandbox.sandbox_timeout
        ):
            return ToolResult(
                success=False,
                error=(
                    "Timeout must be an integer between 1 and "
                    f"{self._settings.sandbox.sandbox_timeout} seconds."
                ),
            )

        # --- Pre-execution checks ---
        violation = self._check_forbidden(code)
        if violation:
            return ToolResult(
                success=False,
                error=f"Sandbox violation: {violation}",
            )

        compiled = self._compile_code(code)
        if isinstance(compiled, ToolResult):
            return compiled  # Compilation error

        try:
            safe_vars = self._validate_vars(vars or {})
            vars_json = json.dumps(safe_vars, ensure_ascii=False)
            if (
                len(vars_json.encode("utf-8"))
                > self._settings.sandbox.max_vars_bytes
            ):
                raise SandboxViolation(
                    "Injected variables exceed the configured byte limit."
                )
            completed = self._run_in_subprocess(
                code=textwrap.dedent(code),
                timeout=effective_timeout,
                variables=safe_vars,
            )
        except SandboxTimeout:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=(
                    "Code execution timed out after "
                    f"{effective_timeout}s."
                ),
                execution_time_ms=elapsed,
            )
        except SandboxViolation as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Sandbox violation during execution: {exc}",
                execution_time_ms=elapsed,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Sandbox process failed: {exc}",
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=(
                    "Sandbox process failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                execution_time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        stdout = str(completed.get("stdout", ""))
        stderr = str(completed.get("stderr", ""))
        max_output = self._settings.sandbox.max_output_length
        truncated = bool(completed.get("truncated", False))
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + "\n... [stdout truncated]"
            truncated = True
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + "\n... [stderr truncated]"
            truncated = True

        return ToolResult(
            success=bool(completed.get("success", False)),
            output=stdout,
            error=completed.get("error"),
            data={
                "stderr": stderr,
                "return_value": completed.get("return_value"),
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

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        allowed_modules = set(self._settings.sandbox.allowed_modules)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return f"Forbidden private name: {node.id}"
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return f"Forbidden private attribute: {node.attr}"
            if (
                isinstance(node, ast.Attribute)
                and node.attr in FORBIDDEN_FILE_API_NAMES
            ):
                return f"File API not allowed: {node.attr}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in allowed_modules:
                        return f"Module not allowed: {root}"
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if not root or root not in allowed_modules:
                    return f"Module not allowed: {root or '<relative>'}"

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

    @staticmethod
    def _validate_vars(variables: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in variables.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise SandboxViolation(f"Invalid variable name: {key!r}")
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise SandboxViolation(
                    f"Variable '{key}' must be JSON-serializable"
                ) from exc
            safe[key] = value
        return safe

    def _run_in_subprocess(
        self,
        code: str,
        timeout: int,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute code in an isolated child process with a hard timeout."""
        trusted_roots = {
            str(Path(sys.base_prefix).resolve()),
            str(Path(sys.prefix).resolve()),
        }
        for path in site.getsitepackages():
            trusted_roots.add(str(Path(path).resolve()))
        user_site = site.getusersitepackages()
        if user_site:
            trusted_roots.add(str(Path(user_site).resolve()))

        payload = json.dumps(
            {
                "code": code,
                "vars": variables,
                "allowed_modules": self._settings.sandbox.allowed_modules,
                "preload_modules": self._extract_preload_modules(code),
                "cpu_seconds": max(1, timeout),
                "memory_bytes": (
                    self._settings.sandbox.memory_mb * 1024 * 1024
                ),
                "max_output_length": (
                    self._settings.sandbox.max_output_length
                ),
                "trusted_read_roots": sorted(trusted_roots),
            }
        )
        sandbox_base = resolve_project_path(
            self._settings.sandbox.temp_dir
        )
        sandbox_base.mkdir(parents=True, exist_ok=True)
        child_env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "PATH",
                "PATHEXT",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "TMP",
                "TEMP",
                "LANG",
                "LC_ALL",
            }
        }
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory(
            prefix="exec-",
            dir=sandbox_base,
        ) as work_dir:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", _SUBPROCESS_RUNNER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=work_dir,
                env=child_env,
            )
            try:
                stdout, stderr = process.communicate(
                    payload,
                    timeout=timeout + 1,
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise SandboxTimeout()
        if process.returncode != 0:
            raise OSError(
                stderr.strip()
                or f"sandbox exited with code {process.returncode}"
            )
        if not stdout:
            raise OSError("sandbox produced no JSON response")
        return json.loads(stdout)

    def _extract_preload_modules(self, code: str) -> list[str]:
        """Return statically declared allow-listed modules in import order."""
        allowed_modules = set(self._settings.sandbox.allowed_modules)
        modules: list[str] = []
        tree = ast.parse(textwrap.dedent(code))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                candidates = [node.module]
            else:
                continue
            for module_name in candidates:
                if (
                    module_name.split(".", 1)[0] in allowed_modules
                    and module_name not in modules
                ):
                    modules.append(module_name)
        return modules


_SUBPROCESS_RUNNER = r"""
import contextlib
import importlib
import io
import json
import os
import sys
import traceback

payload = json.loads(sys.stdin.read())
sys.dont_write_bytecode = True

try:
    import resource
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (payload["cpu_seconds"], payload["cpu_seconds"] + 1),
    )
    resource.setrlimit(
        resource.RLIMIT_AS,
        (payload["memory_bytes"], payload["memory_bytes"]),
    )
except (ImportError, ValueError, OSError):
    pass

trusted_read_roots = [
    os.path.realpath(path)
    for path in payload["trusted_read_roots"]
]

# Import only modules that were statically declared and allow-listed by the
# parent validator. Native dependencies initialize before the audit hook;
# subsequent dynamic-library access from user code remains blocked.
for module_name in payload["preload_modules"]:
    importlib.import_module(module_name)

def is_within_trusted_root(path):
    try:
        resolved = os.path.realpath(os.fspath(path))
    except TypeError:
        return False
    for root in trusted_read_roots:
        try:
            if os.path.commonpath([resolved, root]) == root:
                return True
        except ValueError:
            continue
    return False

def deny_unsafe_audit_event(event, args):
    if event == "open":
        path = args[0]
        if isinstance(path, int):
            return
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        write_flags = (
            getattr(os, "O_WRONLY", 0)
            | getattr(os, "O_RDWR", 0)
            | getattr(os, "O_CREAT", 0)
            | getattr(os, "O_TRUNC", 0)
            | getattr(os, "O_APPEND", 0)
        )
        if (
            any(char in str(mode) for char in "wax+")
            or (isinstance(flags, int) and flags & write_flags)
            or not is_within_trusted_root(path)
        ):
            raise PermissionError("sandbox file access denied")
    elif event in {
        "os.system",
        "os.spawn",
        "os.exec",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.kill",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.mkdir",
        "os.chdir",
        "os.chmod",
        "os.chown",
        "os.truncate",
        "os.link",
        "os.symlink",
        "subprocess.Popen",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "ctypes.dlopen",
        "ctypes.dlsym",
        "ctypes.dlsym/handle",
    }:
        raise PermissionError(f"sandbox operation denied: {event}")

sys.addaudithook(deny_unsafe_audit_event)

allowed_modules = set(payload["allowed_modules"])
real_import = __import__

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("relative imports are disabled")
    root = name.split(".", 1)[0]
    if root not in allowed_modules:
        raise ImportError(f"module not allowed: {root}")
    return real_import(name, globals, locals, fromlist, level)

safe_builtins = {
    "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
    "bool": bool, "bytearray": bytearray, "bytes": bytes, "chr": chr,
    "complex": complex, "dict": dict, "dir": dir, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "hasattr": hasattr,
    "hash": hash, "hex": hex, "id": id, "int": int,
    "isinstance": isinstance, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "ImportError": ImportError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError, "LookupError": LookupError,
    "__import__": safe_import,
}

sandbox_globals = {
    "__builtins__": safe_builtins,
    "__name__": "__sandbox__",
    **payload["vars"],
}

class CappedTextIO(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.length = 0
        self.truncated = False

    def writable(self):
        return True

    def write(self, value):
        text = str(value)
        remaining = max(0, self.limit - self.length)
        if remaining:
            chunk = text[:remaining]
            self.parts.append(chunk)
            self.length += len(chunk)
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def getvalue(self):
        return "".join(self.parts)

stdout_buffer = CappedTextIO(payload["max_output_length"])
stderr_buffer = CappedTextIO(payload["max_output_length"])
success = True
error = None

try:
    compiled = compile(payload["code"], "<sandbox>", "exec")
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exec(compiled, sandbox_globals)
except BaseException as exc:
    success = False
    error = f"{type(exc).__name__}: {exc}"
    traceback.print_exc(file=stderr_buffer)

return_value = sandbox_globals.get("_return")
try:
    json.dumps(return_value)
except (TypeError, ValueError):
    return_value = repr(return_value)

print(json.dumps({
    "success": success,
    "stdout": stdout_buffer.getvalue(),
    "stderr": stderr_buffer.getvalue(),
    "error": error,
    "return_value": return_value,
    "truncated": stdout_buffer.truncated or stderr_buffer.truncated,
}, ensure_ascii=False))
"""
