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


from play_chesscom import MoveExecutor


def test_white_a1_bottom_left():
    box = {'x': 0.0, 'y': 0.0, 'width': 800.0, 'height': 800.0}
    # A1: file=0, rank=0 → white: x=(0+0.5)*100=50, y=(7-0+0.5)*100=750
    x, y = MoveExecutor._square_to_page_coords(chess.A1, flipped=False, board_box=box)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(750.0)


def test_white_h8_top_right():
    box = {'x': 0.0, 'y': 0.0, 'width': 800.0, 'height': 800.0}
    # H8: file=7, rank=7 → white: x=(7+0.5)*100=750, y=(7-7+0.5)*100=50
    x, y = MoveExecutor._square_to_page_coords(chess.H8, flipped=False, board_box=box)
    assert x == pytest.approx(750.0)
    assert y == pytest.approx(50.0)


def test_black_a1_top_right_when_flipped():
    box = {'x': 0.0, 'y': 0.0, 'width': 800.0, 'height': 800.0}
    # A1 flipped: x=(7-0+0.5)*100=750, y=(0+0.5)*100=50
    x, y = MoveExecutor._square_to_page_coords(chess.A1, flipped=True, board_box=box)
    assert x == pytest.approx(750.0)
    assert y == pytest.approx(50.0)


def test_black_h8_bottom_left_when_flipped():
    box = {'x': 0.0, 'y': 0.0, 'width': 800.0, 'height': 800.0}
    # H8 flipped: x=(7-7+0.5)*100=50, y=(7+0.5)*100=750
    x, y = MoveExecutor._square_to_page_coords(chess.H8, flipped=True, board_box=box)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(750.0)


def test_board_offset_is_applied():
    box = {'x': 100.0, 'y': 200.0, 'width': 800.0, 'height': 800.0}
    x, y = MoveExecutor._square_to_page_coords(chess.A1, flipped=False, board_box=box)
    assert x == pytest.approx(150.0)   # 100 + 50
    assert y == pytest.approx(950.0)   # 200 + 750
