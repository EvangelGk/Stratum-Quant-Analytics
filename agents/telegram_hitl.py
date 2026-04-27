"""Telegram Human-in-the-Loop helper.

Sends a message with an optional inline keyboard and polls for the user's
callback-query choice.  Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
in .env (or system env).

Usage (standalone):
    from agents.telegram_hitl import TelegramHITL
    hitl = TelegramHITL()
    choice = hitl.ask("Deploy to production?", ["yes", "no"])
    # choice == "yes" | "no" | None (timeout)
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── env loading ────────────────────────────────────────────────────────────────
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            load_dotenv(env, override=True)
    except ImportError:
        pass


_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramHITL:
    """Send/receive Telegram messages with inline keyboards."""

    def __init__(self, timeout_s: int = 1800) -> None:
        _load_env()
        self.token    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id  = os.getenv("TELEGRAM_CHAT_ID",  "").strip()
        self.timeout_s = timeout_s
        self._offset: int = 0

    def _url(self, method: str) -> str:
        return _BASE.format(token=self.token, method=method)

    def _configured(self) -> bool:
        if not self.token or not self.chat_id:
            log.warning("Telegram not configured (no BOT_TOKEN / CHAT_ID in .env)")
            return False
        return True

    # ── low-level ─────────────────────────────────────────────────────────────
    def send(self, text: str, reply_markup: Optional[dict] = None) -> Optional[int]:
        """Send a message; return message_id or None on failure."""
        if not self._configured():
            return None
        payload: dict = {
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            r = requests.post(self._url("sendMessage"), json=payload, timeout=15)
            r.raise_for_status()
            return r.json().get("result", {}).get("message_id")
        except Exception as exc:
            log.error("Telegram sendMessage failed: %s", exc)
            return None

    def answer_callback(self, callback_query_id: str) -> None:
        try:
            requests.post(
                self._url("answerCallbackQuery"),
                json={"callback_query_id": callback_query_id, "text": "Got it ✅"},
                timeout=10,
            )
        except Exception:
            pass

    def _poll(self) -> list[dict]:
        """Long-poll getUpdates; return list of new updates."""
        try:
            r = requests.get(
                self._url("getUpdates"),
                params={"offset": self._offset, "timeout": 30, "allowed_updates": '["callback_query"]'},
                timeout=35,
            )
            r.raise_for_status()
            updates = r.json().get("result", [])
            if updates:
                self._offset = updates[-1]["update_id"] + 1
            return updates
        except Exception as exc:
            log.debug("getUpdates error: %s", exc)
            return []

    # ── high-level ─────────────────────────────────────────────────────────────
    def ask(
        self,
        text: str,
        choices: list[str],
        timeout_s: Optional[int] = None,
    ) -> Optional[str]:
        """Send *text* with an inline keyboard of *choices*; return chosen string.

        Returns None if the timeout expires without a response.
        """
        if not self._configured():
            print(f"[HITL] Telegram not configured — defaulting to first choice: {choices[0]}")
            return choices[0]

        # Build inline keyboard: each choice is a button in its own row.
        keyboard = {"inline_keyboard": [[{"text": c, "callback_data": c}] for c in choices]}
        msg_id = self.send(text, reply_markup=keyboard)
        if msg_id is None:
            print("[HITL] Could not send Telegram message — defaulting to first choice")
            return choices[0]

        deadline = time.time() + (timeout_s or self.timeout_s)
        print(f"[HITL] Waiting for Telegram response (timeout {timeout_s or self.timeout_s}s) …")

        while time.time() < deadline:
            for update in self._poll():
                cq = update.get("callback_query")
                if not cq:
                    continue
                data = cq.get("data", "")
                if data in choices:
                    self.answer_callback(cq["id"])
                    print(f"[HITL] Received: {data!r}")
                    return data
        print("[HITL] Timeout — no response received")
        return None

    def notify(self, text: str) -> None:
        """Fire-and-forget notification (no reply needed)."""
        self.send(text)
