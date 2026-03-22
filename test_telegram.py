#!/usr/bin/env python
"""
Quick verification tool β€” run this before your next optimizer run to confirm
that Telegram notifications are working correctly.

Usage:
    python test_telegram.py

Requirements:
    1. Create a .env file in the project root with:
           TELEGRAM_BOT_TOKEN=<your bot token from @BotFather>
           TELEGRAM_CHAT_ID=<your personal chat ID>

    2. To get your CHAT_ID:
       - Open Telegram and message your bot once
       - Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
       - Copy the "id" from the "chat" object in the response
"""

import os
import sys
from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)   # override=True: always re-read from file
        print(f"[INFO] Loaded .env from {env_file}")
    else:
        print(f"[WARN] No .env file found at {env_file}")
        print("       Create it with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
except ImportError:
    print("[WARN] python-dotenv not installed β€” reading from system environment only")

import requests

bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

print()
print("=" * 50)
print("Telegram Configuration Check")
print("=" * 50)

if not bot_token:
    print("[FAIL] TELEGRAM_BOT_TOKEN is empty in .env")
    print("       β†’ Open .env and paste the token from @BotFather on Telegram")
    print("       β†’ The token looks like:  123456789:ABCDEFGhijklmnopQRSTUVwxyz")
    sys.exit(1)
else:
    masked = bot_token[:6] + "..." + bot_token[-4:]
    print(f"[OK]  TELEGRAM_BOT_TOKEN = {masked}")

if not chat_id:
    print("[FAIL] TELEGRAM_CHAT_ID is empty in .env")
    print(f"       β†’ Message your bot in Telegram, then open:")
    print(f"         https://api.telegram.org/bot{bot_token}/getUpdates")
    print('         Find: "chat": {"id": YOUR_NUMBER}  and paste that number')
    sys.exit(1)
else:
    print(f"[OK]  TELEGRAM_CHAT_ID   = {chat_id}")

print()
print("Sending test message...")

try:
    resp = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": (
                "β… <b>STRATUM QUANT ANALYTICS β€” Telegram test OK</b>\n"
                "Notifications are configured correctly.\n"
                "You will receive optimizer approval requests here."
            ),
            "parse_mode": "HTML",
        },
        timeout=10,
    )
    if resp.status_code == 200:
        print(f"[OK]  Message sent successfully! (HTTP 200)")
        print("      Check your Telegram app now.")
    else:
        data = resp.json()
        print(f"[FAIL] HTTP {resp.status_code}: {data.get('description', resp.text[:200])}")
        if resp.status_code == 401:
            print("       Your TELEGRAM_BOT_TOKEN is invalid. Get a new one from @BotFather.")
        elif resp.status_code == 400:
            print("       Your TELEGRAM_CHAT_ID may be wrong.")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("[FAIL] Could not connect to Telegram API (no internet?)")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("[FAIL] Request timed out after 10 seconds")
    sys.exit(1)
except Exception as exc:
    print(f"[FAIL] Unexpected error: {exc}")
    sys.exit(1)

print()
print("=" * 50)
print("All checks passed. Notifications will work for the optimizer.")
print("=" * 50)

