# Historical code and configurations

Archived 2026-10-21. Use [the reproduction guide](../reproduce/README.md) for current runs.

| Location | Contents |
| --- | --- |
| `configs/qh9/` | Complete pre-cleanup Hydra tree, including ablations, QHNet baselines, fine-tuning and paired-branch optimization experiments |
| `configs/md17/` | Historical MD17/rMD17 training and fine-tuning configs |
| `legacy/src/dft_process/post_process/` | Original energy/force evaluators, including `measure_energy_force_uma_gpu_v4.py` |
| `legacy/src/md/test_script*/` | Exploratory notebooks, molecular scans, SCF/timing scripts and their checked-in outputs |
| `legacy/scf_march_2026/` | Recovered SCF selection/calculation scripts from the separate packaging checkout / Quasar experiment |
| `legacy/_experiment_scripts/`, `legacy/_mlff_scripts/` | Previous launch examples |
| `legacy/_my_script/`, `legacy/_auto_script/` | Earlier optimization/benchmark helpers |

The [manifest](manifest.json) records each original path, archived path and SHA-256.
Two shell scripts had embedded credentials removed before archival. No datasets,
external run directories, frozen training worktrees or active launchers were moved.

The complete QH9 config snapshot is self-contained, so historical defaults still
compose. From the repository root, explicitly opt into an archived config with:

```bash
PYTHONPATH=src python -m experiment.train_qh9 \
  --config-path ../../archive/configs/qh9 --config-name qhnet --cfg job
```

`--cfg job` only prints the configuration. Remove it only when intentionally
running a historical experiment. `experiment.train_md17` and its fine-tuning
entrypoint now point directly at `archive/configs/md17`; QH9 fine-tuning points
at `archive/configs/qh9`. Legacy scripts retain historical machine-specific
paths and are evidence, not maintained launchers. See [old usage](LEGACY_USAGE.md).
