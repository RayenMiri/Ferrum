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
