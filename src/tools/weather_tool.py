from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class WeatherToolError(Exception):
    """Weather tool exception."""


@dataclass
class WeatherResult:
    location: str
    temperature_celsius: float
    humidity: int
    wind_speed: float
    description: str


class WeatherTool:
    """
    OpenWeatherMap integration.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self) -> None:
        self.api_key = settings.openweather_api_key

    async def execute(
        self,
        location: str,
    ) -> WeatherResult:
        """
        Retrieve weather data.
        """

        if not self.api_key:
            raise WeatherToolError("Missing OpenWeather API key.")

        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",
        }

        async with httpx.AsyncClient(timeout=20) as client:
            try:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

                response.raise_for_status()

                data = response.json()

                result = WeatherResult(
                    location=data["name"],
                    temperature_celsius=data["main"]["temp"],
                    humidity=data["main"]["humidity"],
                    wind_speed=data["wind"]["speed"],
                    description=data["weather"][0]["description"],
                )

                logger.info(
                    "Weather lookup success: %s",
                    location,
                )

                return result

            except Exception as exc:
                logger.exception("Weather lookup failed.")

                raise WeatherToolError(str(exc)) from exc
