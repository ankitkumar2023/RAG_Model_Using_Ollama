from __future__ import annotations

import asyncio

from src.tools.calculator_tool import (
    CalculatorTool,
)
from src.tools.stock_tool import StockTool
from src.tools.weather_tool import (
    WeatherTool,
)
from src.tools.web_search_tool import (
    WebSearchTool,
)


async def calculator_demo() -> None:
    tool = CalculatorTool()

    result = await tool.execute("sqrt(144) + 25 * 2")

    print("\nCalculator Result")
    print(result)


async def weather_demo() -> None:
    tool = WeatherTool()

    result = await tool.execute("London")

    print("\nWeather Result")
    print(result)


async def stock_demo() -> None:
    tool = StockTool()

    result = await tool.execute("AAPL")

    print("\nStock Result")
    print(result)


async def web_search_demo() -> None:
    tool = WebSearchTool()

    results = await tool.execute(
        "Latest AI news",
    )

    print("\nWeb Search Results")

    for result in results:
        print(result)


async def main() -> None:
    await calculator_demo()

    # Uncomment if API keys configured
    # await weather_demo()
    # await stock_demo()
    # await web_search_demo()


if __name__ == "__main__":
    asyncio.run(main())
