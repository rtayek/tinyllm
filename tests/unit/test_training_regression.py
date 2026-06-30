from __future__ import annotations

import os
from pathlib import Path

import pytest

import llm.training_regression as regression_module
from llm.training_regression import (
    clear_regression_run_dir,
    configured_corpora,
    corpora_for_profile,
    parse_args,
    plan_regression_jobs,
)


def test_smoke_profile_selects_small_subset() -> None:
    smoke = corpora_for_profile("smoke")

    assert [corpus.name for corpus in smoke] == [
        "arthur-conan-doyle/adventures-of-sherlock-holmes",
        "lewis-carroll/alices-adventures-in-wonderland",
    ]


def test_all_profile_selects_all_configured_corpora() -> None:
    assert corpora_for_profile("all") == configured_corpora()
    assert len(corpora_for_profile("all")) == 9


def test_regression_jobs_use_regression_run_directories() -> None:
    jobs = plan_regression_jobs("smoke")

    assert jobs
    for job in jobs:
        assert job.runDir.parts[:3] == ("runs", "regression", "smoke")
        assert Path(job.runConfig.trainConfig.runDir or "") == job.runDir
        assert Path(job.runConfig.trainConfig.ckptPath).parent == job.runDir / "checkpoints"
        assert job.runConfig.trainConfig.dataModule == "byte"
        assert job.runConfig.trainConfig.device == "cpu"


def test_invalid_profile_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown training regression profile"):
        corpora_for_profile("missing")


def test_training_regression_parse_args_accepts_profiles() -> None:
    smoke = parse_args(["--profile", "smoke"])
    all_profile = parse_args(["--profile", "all"])

    assert smoke.profile == "smoke"
    assert smoke.no_clean is False
    assert all_profile.profile == "all"


def test_training_regression_parse_args_accepts_no_clean() -> None:
    args = parse_args(["--profile", "smoke", "--no-clean"])

    assert args.no_clean is True


def test_training_regression_parse_args_rejects_invalid_profile() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--profile", "missing"])


@pytest.mark.parametrize(
    "path",
    [
        Path("."),
        Path("runs"),
        Path("runs") / "regression",
        Path("models") / "regression" / "smoke" / "example",
    ],
)
def test_clear_regression_run_dir_refuses_unsafe_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing to delete"):
        clear_regression_run_dir(path)


def test_clear_regression_run_dir_removes_safe_path(tmp_path: Path) -> None:
    run_dir = Path("runs") / "regression" / "smoke" / "example"
    absolute_run_dir = tmp_path / run_dir
    absolute_run_dir.mkdir(parents=True)
    (absolute_run_dir / "artifact.txt").write_text("artifact", encoding="utf-8")

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        clear_regression_run_dir(run_dir)
    finally:
        os.chdir(old_cwd)

    assert not absolute_run_dir.exists()


def test_clear_regression_run_dir_wraps_permission_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(_path: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(regression_module, "_rmtree_with_readonly_retry", fail)
    path = Path("runs") / "regression" / "smoke" / "locked"
    absolute_path = tmp_path / path
    absolute_path.mkdir(parents=True)
    old_cwd = Path.cwd()

    try:
        os.chdir(tmp_path)
        with pytest.raises(PermissionError, match="Explorer, an editor, antivirus"):
            clear_regression_run_dir(path)
    finally:
        os.chdir(old_cwd)
