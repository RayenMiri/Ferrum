import threading

import chess
import pytest
from unittest.mock import AsyncMock, MagicMock

from play_chesscom import HotkeyController


def test_f9_sets_paused():
    ctrl = HotkeyController()
    from pynput.keyboard import Key
    ctrl._on_press(Key.f9)
    assert ctrl.paused.is_set()
    assert not ctrl.quit_flag.is_set()


def test_f10_clears_paused():
    ctrl = HotkeyController()
    ctrl.paused.set()
    from pynput.keyboard import Key
    ctrl._on_press(Key.f10)
    assert not ctrl.paused.is_set()


def test_f11_sets_quit_flag():
    ctrl = HotkeyController()
    from pynput.keyboard import Key
    ctrl._on_press(Key.f11)
    assert ctrl.quit_flag.is_set()


def test_unknown_key_does_nothing():
    ctrl = HotkeyController()
    from pynput.keyboard import Key
    ctrl._on_press(Key.space)
    assert not ctrl.paused.is_set()
    assert not ctrl.quit_flag.is_set()
