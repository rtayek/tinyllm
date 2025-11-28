from __future__ import annotations

import argparse
import sys

from Config import RunConfig
from Model import TinyGpt
from Checkpoints import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage checkpoints (export/load model weights).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-model", help="Export model weights from a checkpoint.")
    export_parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (defaults to config).")
    export_parser.add_argument("--out", type=str, required=True, help="Output path for model weights.")

    load_parser = subparsers.add_parser("load-model", help="Load model weights into a fresh model and save state.")
    load_parser.add_argument("--model", type=str, required=True, help="Path to model-only weights or checkpoint.")
    load_parser.add_argument("--out", type=str, required=True, help="Output path to save loaded model state_dict.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = RunConfig()
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train

    if args.command == "export-model":
        ckpt_mgr = CheckpointManager(model_cfg, train_cfg, trainCkptPath=args.ckpt or None)
        ckpt_mgr.saveModel(args.out)
        print(f"Exported model weights to {args.out}")
    elif args.command == "load-model":
        model = TinyGpt(model_cfg)
        ckpt_mgr = CheckpointManager(model_cfg, train_cfg, modelCkptPath=args.model)
        ckpt_mgr.loadModel(model, args.model)  # pyright: ignore[reportUnknownMemberType]
        # Save the loaded state_dict to the provided output path
        import torch

        torch.save(model.state_dict(), args.out)  # pyright: ignore[reportUnknownMemberType]
        print(f"Loaded model weights from {args.model} and saved state_dict to {args.out}")
    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
