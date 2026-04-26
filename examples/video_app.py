#!/usr/bin/env python

"""Launch the local video pose-to-animation web app."""

from __future__ import annotations

import argparse

from video_app.server import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Animated Drawings video pose app.")
    parser.add_argument("--port", type=int, default=5060, help="Port to listen on. Defaults to 5060.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on. Defaults to 127.0.0.1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
