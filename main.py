from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.inspect_dataset import main as inspect_dataset_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud AI Project")
    parser.add_argument("--inspect-data", action="store_true", help="Print dataset statistics")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.inspect_data:
        inspect_dataset_main()
    else:
        project_root = Path(__file__).resolve().parent
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "5000"))

        print(f"Starting FastAPI frontend from: {project_root}")
        print(f"Open the app at http://{host}:{port}")

        import uvicorn

        uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
