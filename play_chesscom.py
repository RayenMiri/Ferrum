from __future__ import annotations

import argparse
import asyncio
import os
import threading
from pathlib import Path
from typing import Any
from random import uniform
from time import sleep


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
        board_box = await self.page.locator('chess-board, wc-chess-board').first.bounding_box()
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
        # Try wc-chess-board first (new chess.com UI), then chess-board
        for sel in ['wc-chess-board', 'chess-board']:
            loc = self.page.locator(sel).first
            if await loc.count() > 0:
                classes = await loc.get_attribute('class') or ''
                return chess.BLACK if 'flipped' in classes.split() else chess.WHITE
        return chess.WHITE

    async def _get_fen_from_board(self) -> str | None:
        """Extract FEN from chess.com's internal board state via JS."""
        try:
            # wc-chess-board exposes a game object; chess-board uses different internals
            fen = await self.page.evaluate("""() => {
                const board = document.querySelector('wc-chess-board') || document.querySelector('chess-board');
                if (!board) return null;
                // Try multiple APIs chess.com has used
                if (board.game && board.game.getFEN) return board.game.getFEN();
                if (board.getFEN) return board.getFEN();
                if (board.chessboard && board.chessboard.getFEN) return board.chessboard.getFEN();
                return null;
            }""")
            return fen
        except Exception:
            return None

    async def _get_dom_san_list(self) -> list[str]:
        """Try multiple selectors because chess.com changes classes frequently."""
        selectors = [
            'wc-move-list-item .san',
            'wc-move-list .move-text',
            '.move-list .move',
            '.node .san',
            '.move-row .move-san',
            '[data-cy="move-list"] .san',
        ]
        for sel in selectors:
            try:
                elements = await self.page.locator(sel).all()
                if elements:
                    result = []
                    for el in elements:
                        text = await el.inner_text()
                        # Strip move numbers, ratings, annotations
                        cleaned = text.strip()
                        # Remove leading numbers like "1." or "12..."
                        import re
                        cleaned = re.sub(r'^\d+\.+\s*', '', cleaned)
                        cleaned = re.sub(r'[#+!?\\s]', '', cleaned)
                        if cleaned and cleaned not in ('1-0', '0-1', '1/2-1/2', '*'):
                            result.append(cleaned)
                    if result:
                        return result
            except Exception:
                continue
        return []

    async def tick(self, board: chess.Board) -> int:
        # Strategy 1: Try FEN extraction first (most reliable)
        fen = await self._get_fen_from_board()
        if fen:
            try:
                new_board = chess.Board(fen)
                if new_board.fullmove_number >= board.fullmove_number:
                    # Count how many new half-moves
                    new_ply = new_board.ply()
                    old_ply = board.ply()
                    if new_ply > old_ply:
                        # Replay moves from scratch to stay in sync
                        board.set_fen(fen)
                        return new_ply - old_ply
                    return 0
            except ValueError:
                pass

        # Strategy 2: DOM move-list parsing
        san_list = await self._get_dom_san_list()
        already_applied = board.ply()
        
        # Debug: print what we see vs what we expect
        if san_list:
            print(f'[Ferrum] DOM moves: {san_list} | local ply: {already_applied}')

        if already_applied > len(san_list):
            # Local board is ahead of DOM (shouldn't happen, but reset)
            print('[Ferrum] Warning: local board ahead of DOM, resetting')
            board.reset()
            already_applied = 0

        new_moves = san_list[already_applied:]
        applied = 0
        for san in new_moves:
            try:
                board.push_san(san)
                applied += 1
                print(f'[Ferrum] Applied opponent move: {san}')
            except (chess.IllegalMoveError, ValueError) as e:
                print(f'[Ferrum] Failed to parse move "{san}": {e}')
                break
        return applied


class BrowserController:
    def __init__(self, page: Page, username: str, password: str) -> None:
        self.page = page
        self.username = username
        self.password = password

    async def login(self) -> None:
        await self.page.goto('https://www.chess.com/login')
        username_box = self.page.get_by_role('textbox', name='Username, Phone, or Email')
        password_box = self.page.get_by_role('textbox', name='Password')
        login_button = self.page.get_by_role('button', name='Log In')

        await username_box.fill(self.username)
        await password_box.fill(self.password)
        await login_button.click()
        await self.page.wait_for_url('**/home**', timeout=15000)

    async def start_game(self) -> None:
        await self.page.goto('https://www.chess.com/play/online')

        # Wait for the time-control grid to appear instead of networkidle
        await self.page.wait_for_selector(
            'button[data-cy="time-selector-button-180"], .time-selector-button, button:has-text("3 min")',
            timeout=15000
        )

        # Dismiss cookie banner / overlays
        for dismiss_sel in [
            'button[data-cy="cookie-policy-banner-accept"]',
            'button.modal-close-button',
            '[data-cy="close-button"]',
            'button:has-text("Accept all cookies")',
        ]:
            try:
                btn = self.page.locator(dismiss_sel)
                if await btn.count() > 0 and await btn.first.is_visible():
                    await btn.first.click(timeout=3000)
                    await asyncio.sleep(0.3)
            except Exception:
                pass

        # Select 3 min (Blitz)
        for sel in [
            'button[data-cy="time-selector-button-180"]',
            'button:has-text("3 min")',
            '[data-time-control="180"]',
            '.time-selector-button:has-text("3")',
        ]:
            try:
                loc = self.page.locator(sel)
                await loc.first.wait_for(state='visible', timeout=5000)
                await loc.first.click(timeout=5000)
                print('[Ferrum] Selected 3 min time control')
                break
            except Exception:
                pass
        else:
            raise RuntimeError('Could not find 3 min time-control button on chess.com')

        # Click Play / Start Game
        for sel in [
            'button:has(span:text-is("Start Game"))',
            'button:has-text("Start Game")',
            'button:has-text("Play")',
            '[data-cy="new-game-play-button"]',
            '[data-cy="quick-match-button"]',
            'button.play-button',
        ]:
            try:
                loc = self.page.locator(sel)
                await loc.first.wait_for(state='visible', timeout=5000)
                await loc.first.click(timeout=5000)
                print('[Ferrum] Clicked play button')
                break
            except Exception:
                pass
        else:
            raise RuntimeError('Could not find Play/Start Game button on chess.com')

        # Wait for matchmaking — chess.com URL pattern is /game/<id> or /game/live/<id>
        print('[Ferrum] Waiting for opponent...')
        try:
            await self.page.wait_for_url('**/game/**', timeout=90000)
        except Exception:
            # Fallback: just wait for the board to appear regardless of URL
            pass

        print(f'[Ferrum] Current URL: {self.page.url}')

        # Wait for the board element — chess.com uses either chess-board or wc-chess-board
        board_sel = 'chess-board, wc-chess-board'
        await self.page.wait_for_selector(board_sel, timeout=15000)
        print('[Ferrum] Board detected — game started')


async def run(
    browser: BrowserController,
    watcher: BoardWatcher,
    executor: MoveExecutor,
    ferrum: FerrumOpponent,
    hotkeys: HotkeyController,
) -> None:
    await browser.login()

    while not hotkeys.quit_flag.is_set():
        await browser.start_game()

        our_color = await watcher.detect_color()
        flipped = (our_color == chess.BLACK)
        board = chess.Board()

        print(f"[Ferrum] Game started — playing as {'BLACK' if flipped else 'WHITE'}")
        print('[Ferrum] F9=pause  F10=resume  F11=quit')

        # Main game loop
        while not hotkeys.quit_flag.is_set():
            await asyncio.sleep(0.15)
            new_count = await watcher.tick(board)

            if new_count > 0:
                last_san = board.peek().san() if board.move_stack else "?"
                print(f"[Ferrum] Ply {board.ply()} — last move: {last_san} — {'WHITE' if board.turn == chess.WHITE else 'BLACK'} to move")

            if board.is_game_over():
                outcome = board.outcome()
                result_str = outcome.result() if outcome else "unknown"
                print(f"[Ferrum] Game over: {result_str}")
                break

            if board.turn == our_color and not hotkeys.paused.is_set():
                move = ferrum.choose_move(board)
                san = board.san(move)
                sleep(uniform(1, 5))
                board.push(move)
                print(f'[Ferrum] Playing {san}')
                await executor.execute(move, flipped)
                await asyncio.sleep(0.3)

        # Game ended — look for "New 3 min" or "New 1|0" or "Play Again" button
        print('[Ferrum] Looking for rematch/new game button...')
        new_game_clicked = False

        for sel in [
            'button:has(span:text-is("New 3 min"))',
            'button:has-text("New 3 min")',
            'button:has-text("New 1|0")',
            'button:has-text("Play Again")',
            'button:has-text("New Game")',
            '[data-cy="new-game-button"]',
            '[data-cy="new-3-min-button"]',
            'button.new-game-button',
            # 'button:has-text("Rematch")',
        ]:
            try:
                loc = browser.page.locator(sel)
                await loc.first.wait_for(state='visible', timeout=10000)
                await loc.first.click(timeout=5000)
                print(f'[Ferrum] Clicked "{sel}" — starting new game')
                new_game_clicked = True
                # Wait for the board to appear again
                board_sel = 'chess-board, wc-chess-board'
                await browser.page.wait_for_selector(board_sel, timeout=15000)
                print('[Ferrum] New board detected')
                break
            except Exception:
                continue

        if not new_game_clicked:
            print('[Ferrum] No rematch button found — returning to lobby')
            # Fallback: navigate back to play/online and let outer loop restart
            await browser.page.goto('https://www.chess.com/play/online')
            await asyncio.sleep(1)

    print('[Ferrum] Quit flag set — exiting')

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
