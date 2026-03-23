from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HARTREE_TO_EV = 27.211386245988  # eV / Hartree
BOHR_TO_ANG = 0.529177210903  # Angstrom / Bohr
HARTREE_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANG  # eV/Å


@dataclass(frozen=True)
class ResultRow:
    # identifiers
    file_stem: str
    filename: Optional[str]
    status: str  # "ok" | "error"
    error: Optional[str]

    # structure
    atom_count: Optional[int]
    n_H: Optional[int]
    n_heavy: Optional[int]
    min_dist_all_ang: Optional[float]
    min_dist_H_ang: Optional[float]
    max_dist_all_ang: Optional[float]

    # energy errors (from differences)
    energy_err_rks_vs_scflow_eV: Optional[float]
    energy_err_rks_vs_scflow_eV_per_atom: Optional[float]
    energy_err_scf_vs_scflow_eV: Optional[float]
    energy_err_scf_vs_scflow_eV_per_atom: Optional[float]

    # force MAE (overall from differences, and H-only computed from forces arrays)
    forces_mae_rks_vs_scflow_eV_per_ang: Optional[float]
    forces_mae_scf_vs_scflow_eV_per_ang: Optional[float]
    H_force_mae_vec_eV_per_ang: Optional[float]  # mean(||ΔF||) over H atoms
    H_force_mae_comp_eV_per_ang: Optional[float]  # mean(|ΔF_component|) over H atoms & xyz
    hamiltonian_mae_rks_vs_scflow_hartree: Optional[float]


def _compute_min_interatomic_distance_ang(positions_ang: List[List[float]]) -> float:
    """
    positions_ang: list of [x,y,z]
    Returns minimum pairwise distance excluding self.
    """
    pos = positions_ang
    n = len(pos)
    if n < 2:
        return float("nan")
    min_d2 = float("inf")
    for i in range(n - 1):
        xi, yi, zi = pos[i]
        for j in range(i + 1, n):
            xj, yj, zj = pos[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < min_d2:
                min_d2 = d2
    return math.sqrt(min_d2) if min_d2 != float("inf") else float("nan")


def _compute_max_interatomic_distance_ang(positions_ang: List[List[float]]) -> float:
    """
    positions_ang: list of [x,y,z]
    Returns maximum pairwise distance (structure diameter).
    """
    pos = positions_ang
    n = len(pos)
    if n < 2:
        return float("nan")
    max_d2 = 0.0
    for i in range(n - 1):
        xi, yi, zi = pos[i]
        for j in range(i + 1, n):
            xj, yj, zj = pos[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > max_d2:
                max_d2 = d2
    return math.sqrt(max_d2)


def _compute_min_distance_from_mask_ang(
    positions_ang: List[List[float]], mask: List[bool]
) -> float:
    """
    Minimum distance involving at least one masked atom (e.g., H atoms).
    """
    pos = positions_ang
    idx = [i for i, m in enumerate(mask) if m]
    if len(idx) == 0:
        return float("nan")
    n = len(pos)
    if n < 2:
        return float("nan")
    min_d2 = float("inf")
    for i in idx:
        xi, yi, zi = pos[i]
        for j in range(n):
            if j == i:
                continue
            xj, yj, zj = pos[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < min_d2:
                min_d2 = d2
    return math.sqrt(min_d2) if min_d2 != float("inf") else float("nan")


def _force_mae_for_mask_eV_per_ang(
    forces_ref_hartree_per_bohr: List[List[float]],
    forces_pred_hartree_per_bohr: List[List[float]],
    mask: List[bool],
) -> Tuple[float, float]:
    """
    Returns:
      (mean(||ΔF||) over masked atoms, mean(|ΔF_component|) over masked atoms & xyz)
    """
    if len(forces_ref_hartree_per_bohr) != len(forces_pred_hartree_per_bohr):
        raise ValueError("forces length mismatch")
    n = len(forces_ref_hartree_per_bohr)
    if len(mask) != n:
        raise ValueError("mask length mismatch")

    vec_errs: List[float] = []
    comp_abs_sum = 0.0
    comp_abs_n = 0
    for i in range(n):
        if not mask[i]:
            continue
        fr = forces_ref_hartree_per_bohr[i]
        fp = forces_pred_hartree_per_bohr[i]
        dx = (fr[0] - fp[0]) * HARTREE_PER_BOHR_TO_EV_PER_ANG
        dy = (fr[1] - fp[1]) * HARTREE_PER_BOHR_TO_EV_PER_ANG
        dz = (fr[2] - fp[2]) * HARTREE_PER_BOHR_TO_EV_PER_ANG
        vec_errs.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        comp_abs_sum += abs(dx) + abs(dy) + abs(dz)
        comp_abs_n += 3

    if len(vec_errs) == 0:
        return float("nan"), float("nan")
    mae_vec = sum(vec_errs) / len(vec_errs)
    mae_comp = comp_abs_sum / comp_abs_n if comp_abs_n else float("nan")
    return float(mae_vec), float(mae_comp)


def _safe_get(d: Dict[str, Any], path: Iterable[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_result_json(json_path: str | Path) -> ResultRow:
    p = Path(json_path)
    data = json.loads(p.read_text())
    file_stem = p.stem

    filename = data.get("filename")
    atom_count = data.get("atom_count")
    err = data.get("error")
    status = "error" if err else "ok"

    # Structure-level fields
    atomic_numbers = _safe_get(data, ["atom_configuration", "atomic_numbers"])
    positions = _safe_get(data, ["atom_configuration", "positions"])

    atom_count_val: Optional[int] = int(atom_count) if atom_count is not None else None
    n_H: Optional[int] = None
    n_heavy: Optional[int] = None
    min_dist_all: Optional[float] = None
    min_dist_H: Optional[float] = None
    max_dist_all: Optional[float] = None

    if atomic_numbers is not None and positions is not None:
        Z = [int(z) for z in atomic_numbers]
        pos = positions
        h_mask = [z == 1 for z in Z]
        n_H = int(sum(1 for m in h_mask if m))
        if atom_count_val is not None:
            n_heavy = int(atom_count_val - n_H)
        try:
            min_dist_all = _compute_min_interatomic_distance_ang(pos)
            min_dist_H = _compute_min_distance_from_mask_ang(pos, h_mask)
            max_dist_all = _compute_max_interatomic_distance_ang(pos)
        except Exception:
            # keep None if anything unexpected
            min_dist_all = None
            min_dist_H = None
            max_dist_all = None

    # Differences-based summary (already computed in producer)
    energy_err_rks_scflow_eV = _safe_get(data, ["differences", "rks_vs_scflow", "energy", "eV"])
    energy_err_rks_scflow_eV_per_atom = _safe_get(
        data, ["differences", "rks_vs_scflow", "energy", "eV_per_atom"]
    )
    forces_mae_rks_scflow = _safe_get(
        data, ["differences", "rks_vs_scflow", "forces_mae", "eV_per_ang"]
    )
    ham_mae_rks_scflow_hartree = _safe_get(
        data, ["differences", "rks_vs_scflow", "hamiltonian_mae", "hartree"]
    )

    energy_err_scf_scflow_eV = _safe_get(data, ["differences", "scf_vs_scflow", "energy", "eV"])
    energy_err_scf_scflow_eV_per_atom = _safe_get(
        data, ["differences", "scf_vs_scflow", "energy", "eV_per_atom"]
    )
    forces_mae_scf_scflow = _safe_get(
        data, ["differences", "scf_vs_scflow", "forces_mae", "eV_per_ang"]
    )

    # H-only force MAE computed from raw forces
    H_force_mae_vec = None
    H_force_mae_comp = None
    try:
        if status == "ok" and atomic_numbers is not None:
            Z = [int(z) for z in atomic_numbers]
            h_mask = [z == 1 for z in Z]
            # Use RKS as reference; compare vs SCFlow prediction
            rks_forces = _safe_get(data, ["rks", "forces", "hartree_per_bohr"])
            scflow_forces = _safe_get(data, ["scflow", "forces", "hartree_per_bohr"])
            if rks_forces is not None and scflow_forces is not None:
                H_force_mae_vec, H_force_mae_comp = _force_mae_for_mask_eV_per_ang(
                    rks_forces, scflow_forces, h_mask
                )
    except Exception:
        H_force_mae_vec = None
        H_force_mae_comp = None

    return ResultRow(
        file_stem=file_stem,
        filename=filename,
        status=status,
        error=err,
        atom_count=atom_count_val,
        n_H=n_H,
        n_heavy=n_heavy,
        min_dist_all_ang=min_dist_all,
        min_dist_H_ang=min_dist_H,
        max_dist_all_ang=max_dist_all,
        energy_err_rks_vs_scflow_eV=(
            float(energy_err_rks_scflow_eV) if energy_err_rks_scflow_eV is not None else None
        ),
        energy_err_rks_vs_scflow_eV_per_atom=(
            float(energy_err_rks_scflow_eV_per_atom)
            if energy_err_rks_scflow_eV_per_atom is not None
            else None
        ),
        energy_err_scf_vs_scflow_eV=(
            float(energy_err_scf_scflow_eV) if energy_err_scf_scflow_eV is not None else None
        ),
        energy_err_scf_vs_scflow_eV_per_atom=(
            float(energy_err_scf_scflow_eV_per_atom)
            if energy_err_scf_scflow_eV_per_atom is not None
            else None
        ),
        forces_mae_rks_vs_scflow_eV_per_ang=(
            float(forces_mae_rks_scflow) if forces_mae_rks_scflow is not None else None
        ),
        forces_mae_scf_vs_scflow_eV_per_ang=(
            float(forces_mae_scf_scflow) if forces_mae_scf_scflow is not None else None
        ),
        H_force_mae_vec_eV_per_ang=(
            float(H_force_mae_vec) if H_force_mae_vec is not None else None
        ),
        H_force_mae_comp_eV_per_ang=(
            float(H_force_mae_comp) if H_force_mae_comp is not None else None
        ),
        hamiltonian_mae_rks_vs_scflow_hartree=(
            float(ham_mae_rks_scflow_hartree) if ham_mae_rks_scflow_hartree is not None else None
        ),
    )


def load_results_dir(results_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts (one per json).
    """
    d = Path(results_dir)
    json_paths = sorted(d.glob("*.json"))
    rows: List[ResultRow] = [load_result_json(p) for p in json_paths]
    return [r.__dict__ for r in rows]


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    # linear interpolation between closest ranks
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def summarize_values(values: Iterable[Any]) -> Dict[str, float]:
    nums: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isnan(x) or math.isinf(x):
            continue
        nums.append(x)

    if not nums:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    nums.sort()
    mean = sum(nums) / len(nums)
    std = statistics.stdev(nums) if len(nums) > 1 else 0.0
    return {
        "count": float(len(nums)),
        "mean": float(mean),
        "std": float(std),
        "min": float(nums[0]),
        "p25": _quantile(nums, 0.25),
        "p50": _quantile(nums, 0.50),
        "p75": _quantile(nums, 0.75),
        "p95": _quantile(nums, 0.95),
        "max": float(nums[-1]),
    }


def assign_bin(value: Optional[float], edges: List[float]) -> Optional[str]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x):
        return None
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if (x >= lo) and (x < hi):
            return f"[{lo}, {hi})"
    # include last edge if it's inf or value equals last edge
    if x >= edges[-2] and (math.isinf(edges[-1]) or x <= edges[-1]):
        return f"[{edges[-2]}, {edges[-1]})"
    return None


def group_stats(
    rows: List[Dict[str, Any]],
    group_key: str,
    metrics: List[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        k = r.get(group_key)
        groups.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for k in sorted(groups.keys(), key=lambda x: (str(x) if x is not None else "")):
        rs = groups[k]
        entry: Dict[str, Any] = {group_key: k, "n": len(rs)}
        for m in metrics:
            entry[m] = summarize_values([rr.get(m) for rr in rs])["mean"]
            entry[m + "_p50"] = summarize_values([rr.get(m) for rr in rs])["p50"]
        out.append(entry)
    return out

