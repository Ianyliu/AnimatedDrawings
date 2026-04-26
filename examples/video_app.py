#!/usr/bin/env python

"""Launch the local video pose-to-animation web app."""

from __future__ import annotations

import argparse

from video_app.diagnostics import format_diagnostics, run_diagnostics
from video_app.server import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Animated Drawings video pose app.")
    parser.add_argument("--port", type=int, default=5060, help="Port to listen on. Defaults to 5060.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on. Defaults to 127.0.0.1.")
    parser.add_argument("--check", action="store_true", help="Run startup diagnostics and exit.")
    parser.add_argument("--skip-checks", action="store_true", help="Start without running startup diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check or not args.skip_checks:
        checks = run_diagnostics()
        print(format_diagnostics(checks))
        if args.check:
            raise SystemExit(1 if any(check.status == "error" for check in checks) else 0)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
