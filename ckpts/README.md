# QHFlow2 model checkpoints

Model checkpoints are separate distribution artifacts. Source-code licenses do
not automatically determine the license of learned weights, and upstream data
or base-checkpoint terms may still apply.

## Release policy

| Checkpoint family | Count in the linked Drive folder | Release license | Conditions |
| --- | ---: | --- | --- |
| QHFlow2 trained on QH9 Stable/Dynamic | 16 | CC-BY-NC-SA-4.0 | Conservative alignment with QH9; non-commercial, attribution, change indication, and ShareAlike |
| QHFlow2 trained on SchNOrb/MD17 | 6 | MIT | Only for checkpoints trained from scratch without a more restrictive base checkpoint |
| QHFlow2 trained on rMD17 | 6 | MIT | rMD17 source data are CC0; record any additional Hamiltonian-data source |
| QHFlow1 legacy checkpoints | Not fully audited | No license granted yet | Complete provenance review before redistribution |

For the QH9 family, CC-BY-NC-SA-4.0 applies only to checkpoint weight files,
including `.ckpt`, `.pt`, `.pth`, and `.safetensors` files. It does not apply
to QHFlow2 source code, configuration files, this documentation, or checkpoint
metadata. The external QH9 training data independently retains its upstream
license.

The Drive inventory observed on 2026-07-28 contains:

- QH9: four model sizes for each of `QH9Stable-random`,
  `QH9Stable-size_ood`, `QH9Dynamic-300k-geometry`, and
  `QH9Dynamic-300k-mol`;
- MD17: middle and small variants for ethanol, malondialdehyde, and uracil;
- rMD17: middle and small variants for aspirin, naphthalene, and
  salicylic acid.

The existing Drive checkpoint folders contain `.ckpt` files but no individual
license or metadata sidecars. Until the applicable license and metadata travel
with the checkpoint, recipients should not assume a reuse grant.

## Required metadata

Every released checkpoint folder should include a `CHECKPOINT_METADATA.yaml`
with at least:

```yaml
schema_version: 1
name: ""
model_architecture: ""
checkpoint_file: ""
sha256: ""
created_by: "Seongsu Kim"
training_code_commit: ""
base_checkpoint:
  name: null
  source: null
  license: null
datasets:
  - name: ""
    source: ""
    license: ""
license: ""
license_scope: "checkpoint_file_only"
intended_use: "Research on molecular Hamiltonian prediction"
limitations: ""
```

Also include the corresponding full license text from `../LICENSES/`, a link
to the QHFlow2 source commit, training configuration, software versions,
metrics, and required dataset citations.

PyTorch Lightning checkpoints can contain hyperparameters and serialized
objects in addition to weights. Inspect the archive before release and remove
credentials, absolute private paths, W&B secrets, or personal information.
