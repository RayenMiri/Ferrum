from __future__ import annotations

import argparse
import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import chess
from dotenv import load_dotenv
from playwright.async_api import Page, async_playwright

import config
from play_gui import FerrumOpponent, OpponentConfig


class HotkeyController:
    def __init__(self) -> None:
        self.paused: threading.Event = threading.Event()
        self.quit_flag: threading.Event = threading.Event()
        self._listener: Any = None

    def _on_press(self, key: Any) -> None:
        from pynput.keyboard import Key
        try:
            if key == Key.f9:
                self.paused.set()
                print('[Ferrum] Paused (F10 to resume)')
            elif key == Key.f10:
                self.paused.clear()
                print('[Ferrum] Resumed')
            elif key == Key.f11:
                self.quit_flag.set()
                print('[Ferrum] Quitting...')
        except AttributeError:
            pass  # pynput special keys without a Key enum entry

    def start(self) -> None:
        from pynput import keyboard
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


class MoveExecutor:
    def __init__(self, page: Page) -> None:
        self.page = page

    @staticmethod
    def _square_to_page_coords(
        square: int,
        flipped: bool,
        board_box: dict[str, float],
    ) -> tuple[float, float]:
        cell = board_box['width'] / 8.0
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if not flipped:
            x = board_box['x'] + (file_idx + 0.5) * cell
            y = board_box['y'] + (7 - rank_idx + 0.5) * cell
        else:
            x = board_box['x'] + (7 - file_idx + 0.5) * cell
            y = board_box['y'] + (rank_idx + 0.5) * cell
        return x, y
