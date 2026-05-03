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

    async def execute(self, move: chess.Move, flipped: bool) -> None:
        board_box = await self.page.locator('chess-board').bounding_box()
        if board_box is None:
            raise RuntimeError('chess-board element not visible; cannot compute click coordinates')
        from_x, from_y = self._square_to_page_coords(move.from_square, flipped, board_box)
        to_x, to_y = self._square_to_page_coords(move.to_square, flipped, board_box)
        await self.page.mouse.click(from_x, from_y)
        await asyncio.sleep(0.05)
        await self.page.mouse.click(to_x, to_y)
        if move.promotion:
            await asyncio.sleep(0.1)
            await self.page.locator('.promotion-piece').first.click()


class BoardWatcher:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def detect_color(self) -> chess.Color:
        locator = self.page.locator('chess-board')
        classes = await locator.get_attribute('class') or ''
        return chess.BLACK if 'flipped' in classes.split() else chess.WHITE

    async def _get_dom_san_list(self) -> list[str]:
        elements = await self.page.locator('.node .san').all()
        result = []
        for el in elements:
            text = await el.inner_text()
            stripped = text.strip()
            if stripped:
                result.append(stripped)
        return result

    async def tick(self, board: chess.Board) -> int:
        san_list = await self._get_dom_san_list()
        already_applied = board.ply()
        new_moves = san_list[already_applied:]
        applied = 0
        for san in new_moves:
            try:
                board.push_san(san)
                applied += 1
            except (chess.IllegalMoveError, ValueError):
                break  # DOM returned illegal move; re-sync next tick
        return applied


class BrowserController:
    def __init__(self, page: Page, username: str, password: str) -> None:
        self.page = page
        self.username = username
        self.password = password

    async def login(self) -> None:
        await self.page.goto('https://www.chess.com/login')
        await self.page.fill('#username', self.username)
        await self.page.fill('#password', self.password)
        await self.page.click('button[type="submit"]')
        await self.page.wait_for_url('**/home**', timeout=15000)

    async def start_game(self) -> None:
        await self.page.goto('https://www.chess.com/play/online')
        await self.page.click('text=3 min')
        await self.page.click('button.ui-button-primary')
        await self.page.wait_for_selector('chess-board', timeout=60000)
        print('[Ferrum] Board detected — game started')


async def run(
    browser: BrowserController,
    watcher: BoardWatcher,
    executor: MoveExecutor,
    ferrum: FerrumOpponent,
    hotkeys: HotkeyController,
) -> None:
    await browser.login()
    await browser.start_game()

    our_color = await watcher.detect_color()
    flipped = (our_color == chess.BLACK)
    board = chess.Board()

    print(f"[Ferrum] Game started — playing as {'BLACK' if flipped else 'WHITE'}")
    print('[Ferrum] F9=pause  F10=resume  F11=quit')

    while not hotkeys.quit_flag.is_set():
        await asyncio.sleep(0.15)
        new_count = await watcher.tick(board)

        if new_count > 0:
            print(f"[Ferrum] Ply {board.ply()} — {'WHITE' if board.turn == chess.WHITE else 'BLACK'} to move")

        if board.is_game_over():
            print(f"[Ferrum] Game over: {board.outcome().result()}")
            break

        if board.turn == our_color and not hotkeys.paused.is_set():
            move = ferrum.choose_move(board)
            san = board.san(move)
            board.push(move)  # optimistic update — prevents double-play before DOM reflects our move
            print(f'[Ferrum] Playing {san}')
            await executor.execute(move, flipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ferrum chess.com bot')
    parser.add_argument('--checkpoint', required=True, help='Path to Ferrum checkpoint (.pt)')
    parser.add_argument('--username', default=None, help='chess.com username (or set CHESSCOM_USERNAME in .env)')
    parser.add_argument('--password', default=None, help='chess.com password (or set CHESSCOM_PASSWORD in .env)')
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    username = args.username or os.environ.get('CHESSCOM_USERNAME', '')
    password = args.password or os.environ.get('CHESSCOM_PASSWORD', '')

    if not username or not password:
        raise SystemExit(
            'Provide --username/--password or set CHESSCOM_USERNAME/CHESSCOM_PASSWORD in a .env file'
        )

    ferrum = FerrumOpponent(
        OpponentConfig(mode='checkpoint', checkpoint=Path(args.checkpoint))
    )
    hotkeys = HotkeyController()
    hotkeys.start()

    async def _run() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=False)
            page = await browser.new_page()
            ctrl = BrowserController(page, username=username, password=password)
            watcher = BoardWatcher(page)
            executor = MoveExecutor(page)
            try:
                await run(ctrl, watcher, executor, ferrum, hotkeys)
            finally:
                await browser.close()

    try:
        asyncio.run(_run())
    finally:
        hotkeys.stop()
        ferrum.close()


if __name__ == '__main__':
    main()
