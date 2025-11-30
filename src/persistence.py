from __future__ import annotations

import argparse
import sys

from Config import RunConfig
from Model import TinyGpt
from Checkpoint import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage checkpoints (export/load model weights).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    saveParser = subparsers.add_parser("export-model", help="Export model weights from a checkpoint.")
    saveParser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (defaults to config).")
    saveParser.add_argument("--out", type=str, required=True, help="Output path for model weights.")

    loadParser = subparsers.add_parser("load-model", help="Load model weights into a fresh model and save state.")
    loadParser.add_argument("--model", type=str, required=True, help="Path to model-only weights or checkpoint.")
    loadParser.add_argument("--out", type=str, required=True, help="Output path to save loaded model state_dict.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = RunConfig()
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train

    if args.command == "export-model":
        checkpointManager = CheckpointManager(model_cfg, train_cfg, trainCkptPath=args.ckpt or None)
        checkpointManager.saveModel(args.out)
        print(f"Exported model weights to {args.out}")
    elif args.command == "load-model":
        model = TinyGpt(model_cfg)
        checkpointManager = CheckpointManager(model_cfg, train_cfg, modelCkptPath=args.model)
        checkpointManager.loadModel(model, args.model)  # pyright: ignore[reportUnknownMemberType]
        import torch

        torch.save(model.state_dict(), args.out)  # pyright: ignore[reportUnknownMemberType]
        print(f"Loaded model weights from {args.model} and saved state_dict to {args.out}")
    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
