# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Source provenance and compatibility of the additive checkpoint metadata."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import qhflow2_version


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    git(tmp_path, "init")
    source = tmp_path / "src/qhflow2_version.py"
    source.parent.mkdir()
    source.write_text("# tracked source\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=Release test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Test source snapshot")
    monkeypatch.setattr(qhflow2_version, "__file__", str(source))
    return tmp_path


def test_source_metadata_tracks_clean_modified_and_untracked_files(source_repo):
    metadata = qhflow2_version.get_code_metadata()
    assert metadata["git_commit"] == git(source_repo, "rev-parse", "HEAD")
    assert metadata["git_dirty"] is False
    source = source_repo / "src/qhflow2_version.py"
    original = source.read_text()
    source.write_text("# modified source\n")
    assert qhflow2_version.get_code_metadata()["git_dirty"] is True
    source.write_text(original)
    (source_repo / "extra_model.py").write_text("# untracked implementation\n")
    assert qhflow2_version.get_code_metadata()["git_dirty"] is True


def test_archive_does_not_borrow_an_enclosing_repository(source_repo, monkeypatch):
    archive_file = source_repo / "archive/src/qhflow2_version.py"
    archive_file.parent.mkdir(parents=True)
    archive_file.touch()
    monkeypatch.setattr(qhflow2_version, "__file__", str(archive_file))
    metadata = qhflow2_version.get_code_metadata()
    assert metadata["code_version"] == qhflow2_version.__version__
    assert metadata["git_commit"] is None and metadata["git_dirty"] is None


def test_git_is_optional(source_repo, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "check_output", missing_git)
    metadata = qhflow2_version.get_code_metadata()
    assert metadata["code_version"] == qhflow2_version.__version__
    assert metadata["git_commit"] is None and metadata["git_dirty"] is None


@pytest.mark.parametrize("name", ["QHFlow_so2_v5_1_small", "QHFlow2_s", "QHFlow2_O3_s"])
@pytest.mark.parametrize("include_metadata", [False, True])
def test_lightning_checkpoint_loads_with_or_without_release_metadata(tmp_path, monkeypatch, name, include_metadata):
    import torch
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from pl_module.flow_module import LitModel_flow

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config_name = "config_flow_v2_simple" if name.startswith("QHFlow_so2") else "config_stable_muon"
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/config_qh9")):
        conf = compose(config_name=config_name, overrides=[
            f"model={name}", "model.hidden_size=4", "model.bottle_hidden_size=2",
            "model.num_gnn_layers=1", "model.num_ham_gnn_layers=1",
        ])
    old_dtype = torch.get_default_dtype()
    try:
        module = LitModel_flow(conf)
        checkpoint = {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        }
        if include_metadata:
            module.on_save_checkpoint(checkpoint)
            assert checkpoint["qhflow2"]["code_version"] == qhflow2_version.__version__
            assert checkpoint["qhflow2"]["model_version"] == conf.model.version
            assert checkpoint["qhflow2"]["model_name"] == conf.model.model_name
        path = tmp_path / "model.ckpt"
        torch.save(checkpoint, path)
        restored = LitModel_flow.load_from_checkpoint(path, map_location="cpu", weights_only=False)
        assert type(restored.model) is type(module.model)
        assert restored.state_dict().keys() == module.state_dict().keys()
        for key, expected in module.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(old_dtype)
