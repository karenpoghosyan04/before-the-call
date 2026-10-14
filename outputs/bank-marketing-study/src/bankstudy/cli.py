"""Repository-root command line interface."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproducible pre-call Bank Marketing study")
    parser.add_argument("command", choices=["prepare", "run", "report", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/full"))
    parser.add_argument("--name", help="New run directory name; existing runs are never overwritten")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.root / ".mplconfig"))
    if args.command == "prepare":
        from .data import prepare
        print(prepare(args.root)["split_summary"])
    elif args.command == "run":
        from .experiment import run
        config = args.config if args.config.is_absolute() else args.root / args.config
        print(run(args.root, config, args.name))
    else:
        from .report import generate, verify
        directory = args.run_dir if args.run_dir.is_absolute() else args.root / args.run_dir
        if args.command == "report":
            print(generate(directory))
        else:
            verify(directory)
            print("Saved metrics and locked thresholds verified against predictions.")


if __name__ == "__main__":
    main()
