from __future__ import annotations

import pytest

from src.tools.calculator_tool import (
    CalculatorTool,
)


@pytest.mark.asyncio
async def test_calculator_addition() -> None:
    tool = CalculatorTool()

    result = await tool.execute("2 + 2")

    assert result.result == 4


@pytest.mark.asyncio
async def test_calculator_functions() -> None:
    tool = CalculatorTool()

    result = await tool.execute("sqrt(144)")

    assert result.result == 12


@pytest.mark.asyncio
async def test_calculator_complex() -> None:
    tool = CalculatorTool()

    result = await tool.execute("(10 * 5) + 20 / 2")

    assert result.result == 60
