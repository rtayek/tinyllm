import torch

from llm.LRScheduleStrategy import WarmupCosineStrategy


def test_warmup_cosine_schedule_smoke() -> None:
    """
    Short run to assert warmup ramps up and cosine then decays the LR.
    """
    model = torch.nn.Linear(1, 1)
    base_lr = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    strategy = WarmupCosineStrategy(optimizer, max_steps=6, warmup_frac=0.5)

    lrs: list[float] = []
    for _ in range(6):
        optimizer.step()  # type: ignore[reportUnknownMemberType]
        strategy.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    warmup_steps = strategy.warmupSteps
    assert warmup_steps > 1

    # Warmup: non-decreasing LR up to the peak.
    for i in range(warmup_steps - 1):
        assert lrs[i] <= lrs[i + 1] + 1e-9

    peak_lr = max(lrs)
    assert abs(peak_lr - base_lr) < 1e-6

    # Decay: after warmup, LR should not increase and should end below the peak.
    for i in range(warmup_steps, len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9
    assert lrs[-1] < peak_lr
