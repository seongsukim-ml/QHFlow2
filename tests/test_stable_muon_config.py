# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment.train_qh9_stable_muon import validate_config
from experiment.prepare_qh9_stable_muon import MODEL_CONFIGS
from dataset_module.qh9_stable_readonly import QH9StableReadonly


@pytest.mark.parametrize("name", MODEL_CONFIGS)
def test_all_benchmark_configs_are_stable_and_use_requested_rates(name):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/config_qh9")):
        conf = compose(config_name="config_stable_muon", overrides=[f"model={name}"])
    validate_config(conf)
    assert conf.muon.lr == 1e-3 and conf.muon.adamw_lr == 1e-4
    assert conf.warmup_step == 0 and conf.model.use_block_H_fix
    assert conf.partial_val == 1.0
    assert conf.dataset.batch_size * conf.accumulate_grad_batches == 32
    assert conf.dataset.split == "random"
    conf.dataset.split = "size_ood"
    with pytest.raises(ValueError, match="random split"):
        validate_config(conf)
    conf.dataset.split = "random"
    conf.dataset.dataset_name = "QH9Dynamic"
    with pytest.raises(ValueError, match="QH9Stable"):
        validate_config(conf)


def test_readonly_loader_does_not_create_or_reprocess_files(tmp_path):
    processed = tmp_path / "existing"
    shard = processed / "lmdbs/shard_000.lmdb"
    shard.mkdir(parents=True)
    (shard / "data.mdb").touch()  # Only metadata is opened by this test.
    (processed / "ALL_SHARDS_COMPLETED.txt").touch()
    (processed / "index.json").write_text(json.dumps({"index": [[0, i, i] for i in range(3)]}))
    split = tmp_path / "official-split.json"
    split.write_text(json.dumps({"train": [2], "val": [0], "test": [1]}))
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    dataset = QH9StableReadonly(str(processed), str(split), compute_q_tensor=False)
    assert len(dataset) == 3 and dataset.train_mask.tolist() == [2]
    dataset.close()
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    split.write_text(json.dumps({"train": [2], "val": [2], "test": [1]}))
    with pytest.raises(ValueError, match="exactly once"):
        QH9StableReadonly(str(processed), str(split), compute_q_tensor=False)
