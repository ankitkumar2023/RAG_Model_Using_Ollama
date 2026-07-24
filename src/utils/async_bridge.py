from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

_SENTINEL = object()


def to_sync_iterator(
    async_gen_factory: Callable[[], AsyncIterator[T]],
) -> Iterator[T]:
    """
    Bridge an async generator into a synchronous iterator, for use with
    `st.write_stream` (which needs a plain sync iterator, while
    generation/retrieval in this app is async).

    Runs the async generator on a dedicated background thread with its
    own event loop, forwarding items through a queue. `async_gen_factory`
    is a zero-arg callable that *builds* the async generator, rather than
    the generator itself, because an async generator bound to one event
    loop can't safely be driven from a different thread/loop.
    """

    item_queue: queue.Queue = queue.Queue()

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _drain() -> None:
            try:
                async for item in async_gen_factory():
                    item_queue.put((True, item))
            except Exception as exc:
                item_queue.put((False, exc))
            finally:
                item_queue.put(_SENTINEL)

        try:
            loop.run_until_complete(_drain())
        finally:
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = item_queue.get()

        if item is _SENTINEL:
            return

        ok, payload = item

        if not ok:
            raise payload

        yield payload
