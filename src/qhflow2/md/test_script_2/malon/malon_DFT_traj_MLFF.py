#!/usr/bin/env python
"""
Malonaldehyde single-point energy curve from a provided DFT trajectory (no relaxation).
Outputs: malon_energies_single_point.npy, malon_rOH_single_point.npy,
         malon_scan_omat_single_point.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.io import read

try:
    from fairchem.core.common.registry import registry
    from fairchem.core.common.utils import match_state_dict, load_state_dict
    from fairchem.core.datasets import data_list_collater as fairchem_data_list_collater
    from fairchem.core.preprocessing import AtomsToGraphs
    try:
        import fairchem.core.models.equiformer_v2.equiformer_v2  # noqa: F401
        import fairchem.core.models.equiformer_v2.equiformer_v2_dens  # noqa: F401
    except ImportError:
        print("Warning: equiformer_v2 modules import failed.")
except ImportError as exc:
    print(f"Error: fairchem-core import failed: {exc}")
    raise SystemExit(1) from exc

try:
    from fairchem.core.common.registry import registry
    try:
        from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone

        @registry.register_model("equiformer_v2_backbone")
        class EqV2_Wrapper(EquiformerV2Backbone):
            pass

        print("[*] Successfully mapped 'equiformer_v2_backbone' to EquiformerV2Backbone.")
    except ImportError:
        print("[!] Warning: Could not find EquiformerV2Backbone code.")
except Exception as e:
    print(f"[!] Error during model alias registration: {e}")


class FairChemHydraCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, ckpt_path: str, device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_with_mapping(ckpt_path).to(self.device)
        self.model.eval()

        backbone = getattr(self.model, "backbone", None)
        if backbone is not None:
            if hasattr(backbone, "use_pbc"):
                backbone.use_pbc = False
            if hasattr(backbone, "use_pbc_single"):
                backbone.use_pbc_single = False
        self.max_neighbors = getattr(backbone, "max_neighbors", None)
        self.cutoff = getattr(backbone, "cutoff", getattr(backbone, "max_radius", 6.0))
        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors or 200,
            radius=self.cutoff or 6.0,
            r_edges=True,
            r_fixed=True,
            r_pbc=False,
        )

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = atoms.copy()
        atoms.pbc = False
        large_box = np.eye(3) * 50.0
        atoms.set_cell(large_box, scale_atoms=False)
        atoms.center()
        data_object = self.a2g.convert(atoms)
        batch = fairchem_data_list_collater([data_object], otf_graph=True)
        batch = batch.to(self.device)
        with torch.no_grad():
            pred = self.model(batch)

        if "energy" in pred:
            self.results["energy"] = float(pred["energy"].detach().cpu().numpy()[0])
        else:
            self.results["energy"] = np.nan

        if "forces" in pred:
            self.results["forces"] = pred["forces"].detach().cpu().numpy()

    def _load_model_with_mapping(self, ckpt_path: str):
        import inspect

        print(f"[*] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        if "config" in checkpoint:
            model_cfg = dict(checkpoint["config"]["model"])
        else:
            model_cfg = dict(checkpoint["model"])

        state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or {}
        state_for_load = state
        if isinstance(state, dict) and state:
            if all(k.startswith("module.") for k in state.keys()):
                state_for_load = {k[len("module."):]: v for k, v in state.items()}
                print("    [Detect] Stripped 'module.' prefix from state_dict keys.")
            keys = state_for_load.keys()
            eqv2_by_keys = any(k.startswith("backbone.blocks.") for k in keys)
            eqv2_by_keys = eqv2_by_keys or any(k.startswith("backbone.Jd_") for k in keys)
            eqv2_by_keys = eqv2_by_keys or any(
                k.startswith("backbone.edge_degree_embedding") for k in keys
            )
            if eqv2_by_keys and isinstance(model_cfg.get("backbone"), dict):
                model_cfg["backbone"]["model"] = "equiformer_v2_backbone"
                print("    [Detect] Checkpoint matches EquiformerV2.")

        def smart_filter(config_dict, target_class, keep_keys=None):
            if not isinstance(config_dict, dict):
                return
            if keep_keys is None:
                keep_keys = []
            sig = inspect.signature(target_class.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self"}
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            if not has_kwargs:
                keys_to_drop = []
                for k in list(config_dict.keys()):
                    if k not in valid_keys and k not in keep_keys:
                        keys_to_drop.append(k)
                        config_dict.pop(k)
                if keys_to_drop:
                    print(
                        f"    [Auto-Filter] Dropped {len(keys_to_drop)} unused args for "
                        f"{target_class.__name__}: {keys_to_drop}"
                    )

        head_map = {
            "fairchem.core.models.equiformer_v2.heads.EqV2ScalarHead": "equiformer_v2_energy_head",
            "fairchem.core.models.equiformer_v2.heads.EqV2VectorHead": "equiformer_v2_force_head",
            "fairchem.core.models.equiformer_v2.heads.Rank2SymmetricTensorHead": "rank2_symmetric_head",
            "esen_mlp_efs_head_dens": "escn_force_head",
            "fairchem.core.models.esen.heads.esen_mlp_efs_head_dens": "escn_force_head",
        }
        heads = model_cfg.get("heads", {})
        if isinstance(heads, dict):
            for _, cfg in heads.items():
                if not isinstance(cfg, dict):
                    continue
                module_name = cfg.get("module")
                if module_name:
                    if module_name in head_map:
                        cfg["module"] = head_map[module_name]
                    elif module_name.split(".")[-1] in head_map:
                        cfg["module"] = head_map[module_name.split(".")[-1]]

        name = model_cfg.pop("name", None)
        if name is None:
            name = "equiformer_v2_backbone"
        try:
            model_class = registry.get_model_class(name)
        except KeyError:
            print(f"[!] Warning: Model '{name}' not found. Defaulting to equiformer_v2_backbone.")
            model_class = registry.get_model_class("equiformer_v2_backbone")

        keep_meta_keys = ["model", "type", "class", "module"]

        print(f"    -> Filtering top-level config for {model_class.__name__}")
        smart_filter(model_cfg, model_class, keep_keys=keep_meta_keys)

        if "backbone" in model_cfg and isinstance(model_cfg["backbone"], dict):
            try:
                backbone_class = registry.get_model_class("equiformer_v2_backbone")
                print(
                    f"    -> Filtering nested backbone config for {backbone_class.__name__}"
                )
                smart_filter(model_cfg["backbone"], backbone_class, keep_keys=keep_meta_keys)
            except Exception:
                pass

        if isinstance(heads, dict):
            for _, head_cfg in heads.items():
                if not isinstance(head_cfg, dict):
                    continue
                module_key = head_cfg.get("module")
                if module_key:
                    try:
                        head_class = registry.get_model_class(module_key)
                        smart_filter(head_cfg, head_class, keep_keys=keep_meta_keys)
                    except KeyError:
                        pass

        model = model_class(**model_cfg)
        matched = match_state_dict(model.state_dict(), state_for_load)
        load_state_dict(model, matched, strict=False)
        return model


def shift_to_min(arr):
    return arr - np.nanmin(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traj",
        required=True,
        help="Path to DFT trajectory (XYZ/ASE-readable)",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="/home/yoonho/MDsim/dissociation_test/eqV2_31M_omat.pt",
        help="Path to OMAT EquiformerV2 checkpoint",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/yoonho/MDsim/dissociation_test/result_malon_omat_single_point",
        help="Output directory",
    )
    ap.add_argument("--i_Oa", type=int, default=0, help="Acceptor O index")
    ap.add_argument("--i_Od", type=int, default=1, help="Donor O index")
    ap.add_argument("--i_H", type=int, default=5, help="Proton H index")
    ap.add_argument(
        "--plot_prefix",
        type=str,
        default="malon_scan_omat_single_point",
        help="Filename prefix for plot",
    )
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"Trajectory not found: {traj_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    frames = read(str(traj_path), index=":")
    if not frames:
        raise SystemExit(f"No frames read from: {traj_path}")

    calc = FairChemHydraCalculator(
        args.ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    r_scan = []
    energies = []
    for idx, atoms in enumerate(frames):
        atoms.calc = calc
        try:
            r_oh = atoms.get_distance(args.i_Od, args.i_H)
            e = atoms.get_potential_energy()
            r_scan.append(r_oh)
            energies.append(e)
            print(f"{idx:4d} | r(Od-H)={r_oh:8.4f} | E={e:12.6f}")
        except Exception as exc:
            r_scan.append(np.nan)
            energies.append(np.nan)
            print(f"{idx:4d} | Fail ({exc})")

    r_scan = np.array(r_scan)
    energies = np.array(energies)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "malon_rOH_single_point.npy", r_scan)
    np.save(out_dir / "malon_energies_single_point.npy", energies)

    plt.figure(figsize=(8, 6))
    rel = shift_to_min(energies)
    plt.plot(r_scan, rel, "o-", color="navy")
    plt.xlabel("r(O-H) [A]")
    plt.ylabel("Rel. Energy [eV]")
    plt.title("Malonaldehyde Proton Transfer (single point)")
    plt.grid(True, alpha=0.3)
    out_plot = out_dir / f"{args.plot_prefix}.png"
    plt.savefig(out_plot, dpi=200)
    print(f"[*] Done! Saved {out_plot}")


if __name__ == "__main__":
    main()
