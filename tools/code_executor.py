"""Python 代码执行沙箱 - 安全运行 Agent 生成的代码。"""

import ast
import io
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

logger = logging.getLogger(__name__)

# 禁止导入的危险模块
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "importlib", "ctypes", "socket",
    "urllib", "http", "ftplib", "smtplib", "pickle", "marshal",
    "compileall", "py_compile", "bdb", "pdb", "trace",
}

# 禁止在代码中使用的危险函数/方法名（全局禁止 + 模块级禁止）
_FORBIDDEN_CALLS_GLOBAL = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "system", "popen", "call", "run", "exec_",
}
# 以下函数名被导入到当前作用域时也禁止（如 from importlib import import_module）
_FORBIDDEN_CALLS_ALIASED = {
    "import_module", "find_loader",  # importlib
}

# 允许使用的安全模块白名单（如果启用白名单模式）
_ALLOWED_MODULES = {
    "math", "random", "statistics", "datetime", "decimal", "fractions",
    "itertools", "collections", "functools", "heapq", "bisect",
    "json", "re", "string", "textwrap", "hashlib", "base64",
    "typing", "copy", "pprint", "numbers", "abc",
    "numpy", "pandas",
}


class SecurityError(Exception):
    """代码安全检查失败"""
    pass


def _check_ast(code: str) -> bool:
    """通过 AST 检查代码安全性。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"语法错误: {e}")

    for node in ast.walk(tree):
        # 禁止导入
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name.split(".")[0]
                if mod_name in _FORBIDDEN_MODULES:
                    raise SecurityError(f"禁止导入模块: {mod_name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module.split(".")[0] if node.module else ""
            if mod_name in _FORBIDDEN_MODULES:
                raise SecurityError(f"禁止导入模块: {mod_name}")

        # 禁止危险的函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in _FORBIDDEN_CALLS_GLOBAL or fname in _FORBIDDEN_CALLS_ALIASED:
                    raise SecurityError(f"禁止调用函数: {fname}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_CALLS_GLOBAL:
                    raise SecurityError(f"禁止调用方法: {node.func.attr}")
                # 检查模块级危险调用（如 importlib.import_module）
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "importlib" and node.func.attr in _FORBIDDEN_CALLS_ALIASED:
                        raise SecurityError(f"禁止调用: importlib.{node.func.attr}")

        # 禁止 getattr 调用（防范动态属性访问，如 getattr(os, 'system')）
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                raise SecurityError("禁止调用 getattr 以防范动态属性访问")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "getattr":
                raise SecurityError("禁止调用 getattr 以防范动态属性访问")

    return True


def _execute_code_worker(code: str, timeout: int, result_queue: multiprocessing.Queue):
    """在子进程中执行代码（worker 函数）。"""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start_time = time.time()

    try:
        # 重定向 stdout/stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # 创建一个受限的全局命名空间
            import builtins
            safe_builtins = dict(builtins.__dict__)
            for _fname in ("open", "input", "exec", "eval", "compile", "__import__"):
                safe_builtins.pop(_fname, None)
            safe_globals = {
                "__builtins__": safe_builtins,
                "__name__": "__main__",
            }
            exec(code, safe_globals)

        duration_ms = int((time.time() - start_time) * 1000)
        result_queue.put({
            "success": True,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "duration_ms": duration_ms,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        })


def execute_python(code: str, timeout: int = 30) -> dict:
    """安全执行 Python 代码。

    参数:
        code: Python 代码字符串
        timeout: 超时时间（秒）

    返回:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "error": str | None,
            "traceback": str | None,
            "duration_ms": int,
        }
    """
    # 步骤1: AST 安全检查
    try:
        _check_ast(code)
    except SecurityError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": str(e),
            "traceback": "",
            "duration_ms": 0,
        }

    # 步骤2: 在子进程中执行（超时保护）
    # 注意：Windows 上 multiprocessing 需要主模块被 __main__ 保护
    # 这里使用 spawn 启动方式确保子进程独立
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_code_worker,
        args=(code, timeout, result_queue)
    )

    try:
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"代码执行超时（>{timeout}秒）",
                "traceback": "",
                "duration_ms": timeout * 1000,
            }

        if not result_queue.empty():
            return result_queue.get()
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": "执行进程异常退出",
                "traceback": "",
                "duration_ms": 0,
            }

    except Exception as e:
        logger.exception("Code execution failed")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_ms": 0,
        }
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)


def format_result(result: dict) -> str:
    """将执行结果格式化为易读的文本。"""
    lines = []
    if result["success"]:
        lines.append("执行成功 ✅")
    else:
        lines.append("执行失败 ❌")
    if result["stdout"]:
        lines.append("\n[标准输出]")
        lines.append(result["stdout"])
    if result["stderr"]:
        lines.append("\n[标准错误]")
        lines.append(result["stderr"])
    if result["error"]:
        lines.append(f"\n[错误] {result['error']}")
    if result.get("traceback"):
        lines.append("\n[堆栈跟踪]")
        lines.append(result["traceback"])
    lines.append(f"\n耗时: {result.get('duration_ms', 0)}ms")
    return "\n".join(lines)
