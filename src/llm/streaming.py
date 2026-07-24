from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator


async def stream_accumulator(
    stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Accumulate streamed chunks progressively.
    """

    buffer = ""

    async for chunk in stream:
        buffer += chunk
        yield buffer

        await asyncio.sleep(0)
