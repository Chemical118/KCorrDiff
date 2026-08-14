#!/usr/bin/env python3
"""Apply only Telegram credentials from .env without printing secret values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


def _parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, encoded = line.split("=", 1)
        key = key.strip()
        if key not in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}:
            continue
        try:
            values = shlex.split(encoded, comments=True, posix=True)
        except ValueError as error:
            raise ValueError(f"invalid .env quoting at line {line_number}") from error
        value = " ".join(values) if values else ""
        if not value:
            raise ValueError(f"{key} is empty in .env")
        if key in result:
            raise ValueError(f"{key} appears more than once in .env")
        result[key] = value
    missing = {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"} - set(result)
    if missing:
        raise ValueError(f".env lacks required Telegram keys: {sorted(missing)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--namespace", default="ws-md93se6gk3270")
    parser.add_argument("--secret-name", default="kcorrdiff-telegram")
    arguments = parser.parse_args(argv)
    values = _parse_env(arguments.env_file.resolve(strict=True))
    resource = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": arguments.secret_name,
            "namespace": arguments.namespace,
            "labels": {
                "app.kubernetes.io/name": "kcorrdiff",
                "app.kubernetes.io/component": "telegram-notifications",
            },
        },
        "type": "Opaque",
        "stringData": {
            "bot-token": values["TELEGRAM_BOT_TOKEN"],
            "chat-id": values["TELEGRAM_CHAT_ID"],
        },
    }
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(resource).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        # kubectl errors should not contain stringData, but avoid relaying raw
        # output so a future client cannot accidentally echo the request body.
        raise RuntimeError("kubectl failed to apply the Telegram Secret")
    print(
        f"Applied Secret {arguments.namespace}/{arguments.secret_name} "
        "with bot-token and chat-id keys (values hidden)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
