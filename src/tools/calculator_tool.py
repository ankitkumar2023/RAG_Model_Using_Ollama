from __future__ import annotations

import ast
import logging
import math
import operator
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class CalculatorError(Exception):
    """Raised for calculator failures."""


@dataclass
class CalculatorResult:
    expression: str
    result: float | int
    success: bool


class SafeEvaluator(ast.NodeVisitor):
    """
    Safe arithmetic evaluator.
    """

    ALLOWED_OPERATORS: ClassVar[dict] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    ALLOWED_FUNCTIONS: ClassVar[dict] = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": lambda: math.pi,
        "e": lambda: math.e,
    }

    def visit(self, node: ast.AST) -> Any:
        return super().visit(node)

    def evaluate(self, expression: str) -> float:
        parsed = ast.parse(
            expression,
            mode="eval",
        )

        return self.visit(parsed.body)

    def visit_BinOp(
        self,
        node: ast.BinOp,
    ) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)

        operator_type = type(node.op)

        if operator_type not in self.ALLOWED_OPERATORS:
            raise CalculatorError(f"Unsupported operator: {operator_type}")

        return self.ALLOWED_OPERATORS[operator_type](left, right)

    def visit_UnaryOp(
        self,
        node: ast.UnaryOp,
    ) -> Any:
        operand = self.visit(node.operand)

        operator_type = type(node.op)

        if operator_type not in self.ALLOWED_OPERATORS:
            raise CalculatorError("Unsupported unary operator.")

        return self.ALLOWED_OPERATORS[operator_type](operand)

    def visit_Num(
        self,
        node: ast.Num,
    ) -> Any:
        return node.n

    def visit_Constant(
        self,
        node: ast.Constant,
    ) -> Any:
        if not isinstance(node.value, (int, float)):
            raise CalculatorError("Invalid constant type.")

        return node.value

    def visit_Call(
        self,
        node: ast.Call,
    ) -> Any:
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Invalid function call.")

        func_name = node.func.id

        if func_name not in self.ALLOWED_FUNCTIONS:
            raise CalculatorError(f"Function '{func_name}' not allowed.")

        func = self.ALLOWED_FUNCTIONS[func_name]

        args = [self.visit(arg) for arg in node.args]

        return func(*args)

    def generic_visit(
        self,
        node: ast.AST,
    ) -> Any:
        raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")


class CalculatorTool:
    """
    Secure calculation tool.
    """

    def __init__(self) -> None:
        self.evaluator = SafeEvaluator()

    async def execute(
        self,
        expression: str,
    ) -> CalculatorResult:
        """
        Safely evaluate mathematical expression.
        """

        try:
            result = self.evaluator.evaluate(expression)

            logger.info(
                "Calculator executed: %s",
                expression,
            )

            return CalculatorResult(
                expression=expression,
                result=result,
                success=True,
            )

        except Exception as exc:
            logger.exception("Calculation failed.")

            raise CalculatorError(str(exc)) from exc
