from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class StockToolError(Exception):
    """Stock tool exception."""


@dataclass
class StockResult:
    symbol: str
    price: float
    change_percent: str
    volume: str


class StockTool:
    """
    AlphaVantage stock lookup tool.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self.api_key = settings.alpha_vantage_api_key

    async def execute(
        self,
        symbol: str,
    ) -> StockResult:
        """
        Fetch stock quote.
        """

        if not self.api_key:
            raise StockToolError("Missing AlphaVantage API key.")

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        async with httpx.AsyncClient(timeout=20) as client:
            try:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

                response.raise_for_status()

                data = response.json()

                quote = data.get(
                    "Global Quote",
                    {},
                )

                if not quote:
                    raise StockToolError("Invalid stock symbol.")

                result = StockResult(
                    symbol=quote["01. symbol"],
                    price=float(quote["05. price"]),
                    change_percent=quote["10. change percent"],
                    volume=quote["06. volume"],
                )

                logger.info(
                    "Stock lookup success: %s",
                    symbol,
                )

                return result

            except Exception as exc:
                logger.exception("Stock lookup failed.")

                raise StockToolError(str(exc)) from exc
