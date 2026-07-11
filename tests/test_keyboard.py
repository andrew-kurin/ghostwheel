from __future__ import annotations

import ghostwheel.keyboard as keyboard


class FakeFlagsReader:
    argtypes = None
    restype = None

    def __init__(self, flags: int) -> None:
        self.flags = flags

    def __call__(self, _state: int) -> int:
        return self.flags


class FakeCoreGraphics:
    def __init__(self, flags: int) -> None:
        self.CGEventSourceFlagsState = FakeFlagsReader(flags)


def test_macos_shift_pressed_reads_core_graphics_flags(monkeypatch) -> None:
    keyboard._macos_flags_reader.cache_clear()
    monkeypatch.setattr(keyboard.sys, "platform", "darwin")
    monkeypatch.setattr(
        keyboard.ctypes,
        "CDLL",
        lambda _path: FakeCoreGraphics(keyboard.MACOS_SHIFT_FLAG),
    )

    assert keyboard.macos_shift_pressed() is True
    keyboard._macos_flags_reader.cache_clear()


def test_macos_shift_pressed_fails_closed_off_macos(monkeypatch) -> None:
    keyboard._macos_flags_reader.cache_clear()
    monkeypatch.setattr(keyboard.sys, "platform", "linux")

    assert keyboard.macos_shift_pressed() is False
    keyboard._macos_flags_reader.cache_clear()
