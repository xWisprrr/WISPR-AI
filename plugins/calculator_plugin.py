"""Example WISPR plugin: safe arithmetic calculator.

This plugin evaluates simple mathematical expressions using Python's ``ast``
module so that only literal numbers and safe operators are permitted — no
``eval()`` or arbitrary code execution.

Usage
-----
Once discovered by the PluginManager, the plugin can be invoked via:

    result = await plugin_manager.invoke(
        "Calculator",
        task="(12 + 8) * 3.5 / 2",
    )
    # result == "35.0"

Supported operators: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``, ``//``
Supported functions: ``abs``, ``round``
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from plugins.plugin_manager import Plugin


# -- Safe evaluation ----------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs":   abs,
    "round": round,
    "sqrt":  math.sqrt,
    "ceil":  math.ceil,
    "floor": math.floor,
    "log":   math.log,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_fn = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_fn   = _SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        fn = _SAFE_FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id!r}")
        args = [_safe_eval(a) for a in node.args]
        return fn(*args)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _evaluate_expression(expr: str) -> str:
    """Parse and evaluate *expr*, returning the result as a string."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"
    try:
        result = _safe_eval(tree.body)
        # Return integer representation when there is no fractional part
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"


# -- Plugin registration ------------------------------------------------------

async def _handler(task: str, context: Dict[str, Any]) -> str:
    """Evaluate the mathematical expression contained in *task*."""
    return _evaluate_expression(task)


def register() -> Plugin:
    """Entry-point called by PluginManager.discover()."""
    return Plugin(
        name="Calculator",
        version="1.0.0",
        description=(
            "Safe arithmetic calculator. Supports +, -, *, /, **, %, // "
            "and the functions abs(), round(), sqrt(), ceil(), floor(), log()."
        ),
        handler=_handler,
        metadata={"category": "math", "safe_eval": True},
    )
