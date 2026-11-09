"""The main point for importing pytest-asyncio items."""

from __future__ import annotations

from importlib.metadata import version

from .loop_factory_negotiation import (
    LoopFactoryCapability,
    LoopFactoryNegotiationError,
    LoopFactoryNegotiator,
    LoopFactorySnapshot,
)
from .plugin import fixture, is_async_test

__version__ = version(__name__)

__all__ = (
    "fixture",
    "is_async_test",
    "LoopFactoryCapability",
    "LoopFactorySnapshot",
    "LoopFactoryNegotiator",
    "LoopFactoryNegotiationError",
)
