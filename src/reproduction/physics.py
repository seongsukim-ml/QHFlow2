"""Hamiltonian -> occupied orbitals -> one-shot DFT, or paired SCF.

All input matrices use native PySCF AOs, Hartree, and float64. Forces are
the analytic DFT gradient evaluated at supplied orbitals, as in the legacy
evaluation; they are not derivatives through the neural network.
"""
import numpy as np
from scipy.linalg import eigh


def orbitals(h, s, electrons, policy):
    h, s = np.asarray(h, dtype=np.float64), np.asarray(s, dtype=np.float64)
    if h.shape != s.shape or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H and S must be equally sized square matrices")
    if not np.isfinite(h).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite H/S")
    if not np.allclose(h, h.T, atol=1e-8, rtol=0) or not np.allclose(s, s.T, atol=1e-8, rtol=0):
        raise ValueError("Asymmetric H/S")
    if electrons % 2 or not 0 < electrons // 2 <= len(h):
        raise ValueError("This protocol requires an even-electron, closed-shell molecule")

    def solve(tol=None, pad=None):
        if tol is None:
            return eigh(h, s)
        w, u = eigh(s)
        # Preserve the legacy padded-eigenvalue algorithm; this is NOT truncation.
        w = np.where(w > tol, w, tol if pad is None else pad)
        x = u / np.sqrt(w)
        e, c = eigh(x.T @ h @ x)
        return e, x @ c

    mode = policy["mode"]
    if mode == "generalized":
        e, c = solve()
        selected = "generalized"
    elif mode in {"legacy_adaptive", "legacy_fixed"}:
        e, c = solve(policy["pred_tol"], policy["pad_eigval"])
        selected = "padded"
        if mode == "legacy_adaptive":
            safe_e, safe_c = solve(policy["gt_tol"], None)
            # This asymmetric min-based rule reproduces the historical v4 code.
            if abs(np.min((safe_e - e)[:electrons // 2 + 1])) <= policy["switch_threshold_Ha"]:
                e, c, selected = safe_e, safe_c, "safe"
    else:
        raise ValueError(f"Unknown eigensolver policy: {mode}")
    occ = np.zeros(len(e)); occ[:electrons // 2] = 2
    density = (c * occ) @ c.T
    return e, c, occ, density, selected


def build_mf(atoms, positions, config):
    from pyscf import dft, gto
    mol = gto.M(atom=[(int(z), list(p)) for z, p in zip(atoms, positions)],
                basis=config["basis"], unit="Angstrom", charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    if config["density_fit"]:
        mf = mf.density_fit(auxbasis=config.get("auxbasis"))
    if config["backend"] == "gpu":
        # No silent CPU fallback: backend changes must be explicit in the config.
        import gpu4pyscf  # noqa: F401
        mf = mf.to_gpu()
    elif config["backend"] != "cpu":
        raise ValueError("backend must be cpu or gpu")
    mf.xc = config["xc"]
    mf.grids.level = config["grid_level"]
    mf.conv_tol = config["conv_tol"]
    mf.max_cycle = config["max_cycle"]
    mf.init_guess = "minao"
    return mf


def host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def device(value, backend):
    if backend == "gpu":
        import cupy
        return cupy.asarray(value)
    return value


def one_shot(atoms, pos, h, s, config, policy):
    mf = build_mf(atoms, pos, config)
    if not np.allclose(host(mf.get_ovlp()), s, atol=2e-6, rtol=0):
        raise ValueError("Overlap does not match geometry/native PySCF AO convention")
    e, c, occ, density, selected = orbitals(h, s, int(sum(atoms)), policy)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = [device(x, config["backend"]) for x in (e, c, occ)]
    energy = float(mf.energy_tot(device(density, config["backend"])))
    grad = mf.nuc_grad_method()
    grad.grid_response = config["grid_response"]
    if config["density_fit"]:
        grad.auxbasis_response = config["auxbasis_response"]
    forces = -host(grad.kernel(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ))
    return dict(energy_Ha=energy, forces_Ha_per_Bohr=forces.tolist(),
                eigensolver_branch=selected, electron_trace=float(np.trace(density @ s)))


def solve_scf(atoms, pos, density, config):
    import time
    mf = build_mf(atoms, pos, config)
    history = []
    def record(env):
        history.append({"cycle": int(env["cycle"]) + 1, "energy_Ha": float(env["e_tot"])})
    mf.callback = record
    started = time.perf_counter()
    energy = mf.kernel(dm0=None if density is None else device(density, config["backend"]))
    return dict(energy_Ha=float(energy), converged=bool(mf.converged), cycles=len(history),
                history=history, wall_seconds=time.perf_counter() - started)
