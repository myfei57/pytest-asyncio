"""The main point for importing pytest-asyncio items."""

from __future__ import annotations

from importlib.metadata import version

from .plugin import (
    COROUTINE_WRAPPER_VERSION,
    CoroutineWrapperError,
    CoroutineWrapperSource,
    fixture,
    get_coroutine_wrapper_chain,
    is_async_test,
)

__version__ = version(__name__)

__all__ = (
    "COROUTINE_WRAPPER_VERSION",
    "CoroutineWrapperError",
    "CoroutineWrapperSource",
    "fixture",
    "get_coroutine_wrapper_chain",
    "is_async_test",
)
