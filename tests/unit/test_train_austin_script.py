import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]


def _write_fake_python(bin_dir: Path, log_path: Path) -> None:
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_PYTHON_ARGS\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_python.chmod(0o755)


def _script_env(tmp_path: Path, run_dir: Path, args_log: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, args_log)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_PYTHON_ARGS"] = str(args_log)
    env["RUN_DIR"] = str(run_dir)
    return env


def test_train_austin_resume_mode_preserves_files_and_forwards_args(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    metrics = run_dir / "metrics.jsonl"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    args_log = tmp_path / "python-args.txt"
    env = _script_env(tmp_path, run_dir, args_log)
    env["RESUME"] = "1"
    env["MAX_STEPS"] = "7000"

    subprocess.run(
        [
            "sh",
            "train-austin.sh",
            "--reset-early-stopping",
            "--early-stop-patience",
            "10",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    assert checkpoint.read_text(encoding="utf-8") == "checkpoint"
    assert metrics.read_text(encoding="utf-8") == "metrics"
    args = args_log.read_text(encoding="utf-8").splitlines()
    assert "--max-steps" in args
    assert args[args.index("--max-steps") + 1] == "7000"
    assert "--reset-early-stopping" in args
    assert args[args.index("--early-stop-patience") + 1] == "10"


def test_train_austin_clean_mode_removes_previous_run_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    metrics = run_dir / "metrics.jsonl"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    args_log = tmp_path / "python-args.txt"
    env = _script_env(tmp_path, run_dir, args_log)
    env["RESUME"] = "0"

    subprocess.run(
        ["sh", "train-austin.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    assert not checkpoint.exists()
    assert not metrics.exists()


def test_train_austin_defaults_to_5000_max_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    args_log = tmp_path / "python-args.txt"
    env = _script_env(tmp_path, run_dir, args_log)

    subprocess.run(
        ["sh", "train-austin.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    args = args_log.read_text(encoding="utf-8").splitlines()
    assert "--max-steps" in args
    assert args[args.index("--max-steps") + 1] == "10000"
