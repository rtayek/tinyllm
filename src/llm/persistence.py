from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, cast

import torch

from .Config import RunConfig
from .Model import TinyGPTLanguageModel
from .Checkpoint import Checkpoint


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
    model_cfg = run_cfg.modelConfig
    train_cfg = run_cfg.trainConfig

    if args.command == "export-model":
        checkpoint_path = args.ckpt or train_cfg.ckptPath
        checkpoint = Checkpoint.load(checkpoint_path, train_cfg.device)
        checkpoint.exportModel(args.out)
        print(f"Exported model weights to {args.out}")
    elif args.command == "load-model":
        state = torch.load(  # pyright: ignore[reportUnknownMemberType]
            args.model,
            map_location=train_cfg.device,
            weights_only=True,
        )
        model_state: Dict[str, Any]
        if isinstance(state, dict) and "modelState" in state:
            model_state = cast(Dict[str, Any], state["modelState"])
        else:
            model_state = cast(Dict[str, Any], state)

        model = TinyGPTLanguageModel(model_cfg)
        model.load_state_dict(model_state)
        torch.save(model.state_dict(), args.out)  # pyright: ignore[reportUnknownMemberType]
        print(f"Loaded model weights from {args.model} and saved state_dict to {args.out}")
    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
