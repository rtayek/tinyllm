from __future__ import annotations
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
