"""Platform keyboard-state fallbacks for ambiguous terminal input."""

from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from typing import Any

MACOS_SHIFT_FLAG = 0x0002_0000
MACOS_COMBINED_SESSION_STATE = 0
CORE_GRAPHICS_PATH = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"


@lru_cache(maxsize=1)
def _macos_flags_reader() -> tuple[ctypes.CDLL, Any] | None:
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL(CORE_GRAPHICS_PATH)
        reader = library.CGEventSourceFlagsState
        reader.argtypes = [ctypes.c_int]
        reader.restype = ctypes.c_uint64
    except (AttributeError, OSError):
        return None
    return library, reader


def macos_shift_pressed() -> bool:
    """Read the current macOS Shift state when a terminal drops modifiers."""

    resolved = _macos_flags_reader()
    if resolved is None:
        return False
    _library, reader = resolved
    try:
        flags = int(reader(MACOS_COMBINED_SESSION_STATE))
    except (OSError, TypeError, ValueError):
        return False
    return bool(flags & MACOS_SHIFT_FLAG)
