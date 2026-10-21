"""Scientifically meaningful checks for the public replay protocol."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.metric import cal_orbital_and_energies
from experiment.train_release import evaluation_indices
from reproduction.physics import build_mf, one_shot, orbitals, solve_scf
from reproduction.qh9 import evaluate, prepare, read_cohort, sha, summarize, write_json
from reproduction.scf_metrics import aggregate_cycles, summarize_recorded

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size,devices,micro,accum", [("s",1,32,1),("o3_s",1,32,1),("m",1,32,1),("l",2,16,1),("xl",4,2,4)])
def test_release_training_profiles(size, devices, micro, accum):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/config_qh9")):
        c = compose(config_name="config_release_" + size)
    assert (c.devices, c.dataset.batch_size, c.accumulate_grad_batches) == (devices,micro,accum)
    assert devices * micro * accum == 32
    assert c.muon.lr == 1e-3 and c.muon.adamw_lr == 1e-4
    assert c.model.num_gnn_layers == {"s":3,"o3_s":3,"m":3,"l":4,"xl":5}[size]
    assert not c.model.get("batch_branches",False)


@pytest.mark.parametrize("length", [1, 13, 13083, 13084])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_eval_shards_cover_every_molecule_once(length, world):
    values = [i for rank in range(world) for i in evaluation_indices(length, rank, world)]
    assert sorted(values) == list(range(length))


def test_recovered_cohort_pairs_all_models():
    rows = read_cohort(ROOT / "reproduce/scf/recorded-300.csv")
    expected = sorted(np.random.default_rng(42).choice(13084, 300, replace=False).tolist())
    assert [r["prediction_index"] for r in rows] == expected
    assert all(float(r["max_position_error_Angstrom"]) < 2e-6 for r in rows)
    with (ROOT / "reproduce/scf/recorded-300-cycles.csv").open() as f:
        cycles = list(csv.DictReader(f))
    assert len(cycles) == 1800
    for model in {r["model"] for r in cycles}:
        assert {int(r["dataset_index"]) for r in cycles if r["model"] == model} == {r["dataset_index"] for r in rows}


def test_cycle_ratio_and_eric_have_distinct_aggregation_and_overhead():
    # Unequal denominators distinguish a ratio of sums from a mean of ratios.
    m = aggregate_cycles([2, 8], [4, 16])
    assert m["cycle_ratio_percent"] == 50
    assert m["cycle_reduction_percent"] == 50
    assert m["eric"] == pytest.approx((3 / 4 + 9 / 16) / 2)
    rows = [dict(dataset_index=i, status="ok", solves={
        "minao": dict(converged=True, cycles=b),
        "predicted": dict(converged=True, cycles=n),
        "dataset_hamiltonian": dict(converged=True, cycles=1),
    }) for i, (n, b) in enumerate(zip([2, 8], [4, 16]))]
    assert summarize(rows, "scf")["eric"] == m["eric"]
    rows[0]["solves"]["predicted"]["converged"] = False
    assert "eric" not in summarize(rows, "scf")


@pytest.mark.parametrize("model,baseline", [([], []), ([1], [0]), ([1], [2, 3]), ([-1], [2]), ([1.5], [2])])
def test_cycle_metrics_reject_invalid_pairs(model, baseline):
    with pytest.raises(ValueError):
        aggregate_cycles(model, baseline)


def test_recorded_scf_summary_matches_source_and_rejects_missing_pairs(tmp_path):
    cohort = ROOT / "reproduce/scf/recorded-300.csv"
    cycles = ROOT / "reproduce/scf/recorded-300-cycles.csv"
    summary = summarize_recorded(cohort, cycles)
    assert summary == json.loads((ROOT / "reproduce/scf/recorded-300-summary.json").read_text())
    evidence = json.loads((ROOT / "reproduce/scf/provenance.json").read_text())["evidence"]
    assert len(summary["models"]) == 6
    for label, value in summary["models"].items():
        assert value["count"] == 300
        assert value["cycle_reduction_percent"] == pytest.approx(evidence[label]["summary"]["cycle_reduction_pct"])
    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("\n".join(cycles.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing, duplicate or mismatched"):
        summarize_recorded(cohort, incomplete)


@pytest.mark.parametrize("mode", ["legacy_fixed", "legacy_adaptive"])
def test_padded_orbitals_match_legacy_torch_formula(mode):
    rng = np.random.default_rng(42)
    u, _ = np.linalg.qr(rng.normal(size=(6,6)))
    s = u @ np.diag([1e-5, .1, .2, .8, 1.2, 2.]) @ u.T
    h = rng.normal(size=(6,6)); h = (h + h.T)/2
    p = dict(mode=mode,pred_tol=1e-3,gt_tol=1e-8,pad_eigval=1.,switch_threshold_Ha=.03)
    e,c,occ,d,branch = orbitals(h,s,4,p)
    tol,pad = (1e-8,None) if branch == "safe" else (1e-3,1.)
    te,tc = cal_orbital_and_energies(torch.tensor(s)[None],torch.tensor(h)[None],tol=tol,pad_eigval=pad)
    np.testing.assert_allclose(e,te[0],rtol=1e-8,atol=1e-8)
    ref=2*tc[0,:,:2]@tc[0,:,:2].T
    np.testing.assert_allclose(d,ref,rtol=1e-8,atol=1e-8)


def tiny_config():
    return dict(backend="cpu",xc="b3lyp",basis="def2-svp",density_fit=True,auxbasis=None,
                grid_level=0,grid_response=False,auxbasis_response=True,conv_tol=1e-11,max_cycle=50)


def test_oneshot_matches_converged_dft_energy_and_analytic_force():
    from pyscf import lib
    lib.num_threads(1)
    a=np.array([1,1]);p=np.array([[0.,0.,0.],[0.,0.,.74]])
    cfg=tiny_config();mf=build_mf(a,p,cfg);mf.kernel()
    assert mf.converged
    h=mf.get_fock();s=mf.get_ovlp()
    result=one_shot(a,p,h,s,cfg,dict(mode="generalized"))
    assert result["energy_Ha"] == pytest.approx(mf.e_tot,abs=1e-9)
    np.testing.assert_allclose(result["forces_Ha_per_Bohr"],-mf.nuc_grad_method().kernel(),atol=2e-7)
    assert result["electron_trace"] == pytest.approx(2.,abs=1e-9)
    solved=solve_scf(a,p,orbitals(h,s,2,dict(mode="generalized"))[3],cfg)
    assert solved["converged"] and solved["cycles"] > 0


def test_failed_or_nonconverged_samples_do_not_produce_headline_average():
    rows=[dict(dataset_index=1,status="error")]
    assert not summarize(rows,"energy_force")["publication_ready"]
    row=dict(dataset_index=1,status="ok",solves={x:dict(converged=False,cycles=50) for x in ["minao","predicted","dataset_hamiltonian"]})
    s=summarize([row],"scf")
    assert not s["publication_ready"] and "cycle_ratio_percent" not in s


def test_prepare_checks_identity_and_preserves_saved_reference(monkeypatch, tmp_path):
    from dataset_module import qh9_stable_readonly
    atoms=torch.tensor([[1],[1]]);pos=torch.tensor([[0.,0.,0.],[0.,0.,.74]],dtype=torch.float64)
    s=torch.eye(2,dtype=torch.float64);h=torch.diag(torch.tensor([-1.,.5],dtype=torch.float64))
    gt=dict(atoms=atoms,pos=pos,overlap_matrix=s,hamiltonian=h)
    class Dataset:
        test_mask=torch.tensor([17])
        def __init__(self,*args,**kwargs): pass
        def get_gt_outputs(self,idx):
            assert idx==17
            return gt
        def close(self): pass
    monkeypatch.setattr(qh9_stable_readonly,"QH9StableReadonly",Dataset)
    (tmp_path/"cohort.csv").write_text("prediction_index,dataset_index\n4,17\n")
    (tmp_path/"split.json").write_text('{"test":[17]}')
    (tmp_path/"weights.ckpt").write_bytes(b"checkpoint association fixture")
    saved=dict(atoms=atoms,pos=pos,overlap=s,format="pyscf_def2svp",length_unit="angstrom")
    torch.save(dict(pred_hamiltonian=h,**saved),tmp_path/"pred_4.pt")
    reference=h.float()+.01
    torch.save(dict(hamiltonian=reference,**saved),tmp_path/"gt_4.pt")
    args=SimpleNamespace(cohort=tmp_path/"cohort.csv",pred_dir=tmp_path,gt_dir=tmp_path,
                         checkpoint=tmp_path/"weights.ckpt",processed_directory=tmp_path,
                         split_file=tmp_path/"split.json",output=tmp_path/"prepared")
    prepare(args)
    with np.load(args.output/"sample_000017.npz") as x:
        np.testing.assert_array_equal(x["H_reference"],reference.double().numpy())
    saved['pos']=pos+1
    torch.save(dict(pred_hamiltonian=h,**saved),tmp_path/"pred_4.pt")
    args.output=tmp_path/"wrong_identity"
    with pytest.raises(ValueError,match="geometry does not match"):
        prepare(args)


def test_resume_rejects_mutated_inputs_and_config(tmp_path):
    import yaml
    cfg=tiny_config();a=np.array([1,1]);p=np.array([[0.,0.,0.],[0.,0.,.74]])
    mf=build_mf(a,p,cfg);mf.kernel()
    np.savez(tmp_path/"sample.npz",atoms=a,pos_ang=p,H_pred=mf.get_fock(),H_reference=mf.get_fock(),S=mf.get_ovlp())
    manifest=dict(matrix_convention="native_pyscf",position_unit="Angstrom",matrix_unit="Hartree",
                  records=[dict(dataset_index=0,prediction_index=0,file="sample.npz",sha256=sha(tmp_path/"sample.npz"))])
    write_json(tmp_path/"manifest.json",manifest)
    config=dict(mode="energy_force",threads=1,dft=cfg,orbitals=dict(mode="generalized"),reference_orbitals=dict(mode="generalized"))
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(config))
    args=SimpleNamespace(config=tmp_path/"config.yaml",manifest=tmp_path/"manifest.json",output=tmp_path/"output")
    evaluate(args)
    result=args.output/"sample_000000.json";before=result.stat().st_mtime_ns
    evaluate(args)
    assert result.stat().st_mtime_ns==before
    assert json.loads((args.output/"summary.json").read_text())["energy_mae_Ha"]==0
    with (tmp_path/"sample.npz").open("ab") as f:f.write(b"changed")
    with pytest.raises(ValueError,match="checksum"):
        evaluate(args)
    config["dft"]["grid_level"]=1
    args.config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match="different inputs/settings"):
        evaluate(args)
