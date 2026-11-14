"""Minimalist web-server entrypoint (`dkmodel`) for the DraftKings Predictions Web UI."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from src.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="dkmodel",
        description="Launch the DraftKings Predictions (DKeX) NFL & NBA Web Application.",
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help="Bind host address (default: $HOST or 0.0.0.0 for remote/network access).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=settings.port,
        help="Bind port (default: $PORT or 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    return parser


def app(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in ("serve", "dashboard"):
        raw_args = raw_args[1:]

    args = build_parser().parse_args(raw_args)
    settings = get_settings()
    auth_note = "passcode protected" if settings.app_passcode else "open access (set APP_PASSCODE to lock)"
    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host

    print(
        f"DKeX Predictions UI running on http://{args.host}:{args.port} "
        f"(local: http://{display_host}:{args.port}) • {auth_note}"
    )
    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    app()
