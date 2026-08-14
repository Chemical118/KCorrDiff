#!/usr/bin/env python3
"""Send a Telegram message using credentials supplied through the environment."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence
from urllib import parse, request


def notify(message: str, *, optional: bool = False) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        if optional:
            return
        raise RuntimeError("Telegram credentials are missing")
    payload = parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    with request.urlopen(request.Request(endpoint, data=payload), timeout=20) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Telegram API rejected the notification")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message")
    parser.add_argument("--optional", action="store_true")
    arguments = parser.parse_args(argv)
    notify(arguments.message, optional=arguments.optional)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
