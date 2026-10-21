"""The public presets must activate the runtime measured by the benchmarks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from common.compile import compile_from_config
from common import training_utils
from experiment.train_qh9_stable_muon import validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,paired", [("config_stable_muon_optimized", False),
                                        ("config_stable_muon_optimized_o3", True)])
def test_presets_activate_compilation_and_versioned_validation(name, paired):
    config_dir = ROOT / ("archive/configs/qh9" if paired else "src/config_qh9")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        conf = compose(config_name=name)
    validate_config(conf)
    assert conf.inference_mode is False
    assert conf.compile.scope == "blocks" and conf.compile.mode == "default"
    assert conf.model.get("batch_branches", False) == paired
    assert conf.dataset.batch_size * conf.accumulate_grad_batches == 32
    if paired:
        from models import get_model
        conf.model.hidden_size = 4
        conf.model.bottle_hidden_size = 2
        conf.model.num_gnn_layers = 1
        conf.model.num_ham_gnn_layers = 1
        assert get_model(conf.model).batch_branches


@pytest.mark.parametrize("inference_mode", [True, False])
def test_trainer_honors_inference_mode_without_changing_legacy_default(monkeypatch, tmp_path, inference_mode):
    mock = Mock()
    monkeypatch.setattr(training_utils.pl, "Trainer", mock)
    conf = OmegaConf.create({} if inference_mode else {"inference_mode": False})
    training_utils.setup_trainer(conf, [], [], tmp_path)
    assert mock.call_args.kwargs["inference_mode"] is inference_mode


def test_compile_config_uses_in_place_modules_and_is_explicit(monkeypatch):
    import common.compile as compilation
    call = Mock()
    monkeypatch.setattr(compilation, "compile_flow", call)
    module = object()
    compile_from_config(module, OmegaConf.create({}))
    call.assert_not_called()
    compile_from_config(module, OmegaConf.create({"compile": {"scope": "blocks", "mode": "default"}}))
    call.assert_called_once_with(module, scope="blocks", backend="inductor", mode="default")


def test_training_entrypoint_compiles_replaced_backbone_before_fit(monkeypatch, tmp_path):
    from experiment import train_qh9
    events = []
    new_backbone = SimpleNamespace()
    module = SimpleNamespace(model=object())
    replacement = SimpleNamespace(load_from_checkpoint=lambda *a, **k: SimpleNamespace(model=new_backbone))
    monkeypatch.setattr(train_qh9, "_get_model_by_pl_type", lambda _: replacement)
    def compile_check(mod, conf):
        assert mod.model is new_backbone
        events.append("compiled")
    monkeypatch.setattr(train_qh9, "compile_from_config", compile_check)
    trainer = SimpleNamespace(fit=lambda *a, **k: events.append("fit"), test=lambda *a, **k: None)
    conf = OmegaConf.create({"use_pretrained_model": True, "predict_after_train": False})
    train_qh9._run_qh9_training_or_testing("train", trainer, module, None, None, None,
                                         None, "weights.ckpt", conf, tmp_path)
    assert events == ["compiled", "fit"]
