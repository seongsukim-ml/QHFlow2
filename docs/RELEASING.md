# Code versions and releases

## Selecting code and a model

The code version lives in **`src/qhflow2_version.py`**. New releases use
`vMAJOR.MINOR.PATCH` tags, with `-rc.N` for release candidates. The existing
`ver.0.1.0` tag keeps its original name and contents. This follows
[Semantic Versioning](https://semver.org/); the project is still in the 0.x
development series.

`main` contains the maintained source, including both legacy and corrected
model classes. Development branches are working areas, not release identifiers.
Pin a published tag when reproducing an experiment:

```bash
cd /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-equivariance-fix
git fetch origin --tags
git switch --detach v0.2.0-rc.1
python3 src/qhflow2_version.py --json
```

The source version and model configuration serve different purposes:

| Identifier | Example | Meaning |
| --- | --- | --- |
| Code release | `v0.2.0-rc.1` | Source, training code, and configurations at a fixed commit |
| Architecture (`model.version`) | `QHFlow2_order_fixed` or `QHFlow2_O3` | SO(3) or identity/inversion-averaged O(3) model |
| Hydra model config | `QHFlow2_s` or `QHFlow2_O3_s` | Architecture, size, and model settings |
| Checkpoint | A particular training run/step | Learned weights, optimizer state, and run configuration |

Older model configuration names remain supported. Pinning a new code release
does not automatically convert old weights to a corrected architecture. Keep
legacy weights with their original model config. The packing-only model
`QHFlow_so2_v5_1_ordering_fixed` remains a historical intermediate variant.

## Runtime and checkpoint identity

From `src/`, use `python3 -m qhflow2_version` or:

```python
from qhflow2_version import __version__, get_code_metadata
print(__version__)
print(get_code_metadata())
```

New Lightning checkpoints have a top-level `qhflow2` dictionary containing
`code_version`, `release_tag`, `git_commit`, `git_dirty`, `model_version`, and
`model_name`. It describes the source loaded at the start of the saving run.
The same source fields appear in CPU preparation reports and training logs.
Model state-dict keys are unchanged, and old checkpoints without this metadata
remain loadable.

`release_tag` is the declared version label; `git_commit` and `git_dirty`
distinguish a tagged checkout from later edits. Git fields are `null` when
unavailable, including source ZIP/tar archives. The version API works without
PyTorch or Git installed. Snapshot or pin the checkout before training; editing
source files while a run is active is not a supported way to version a run.

## Publishing the next version

1. Update `__version__` in `src/qhflow2_version.py`. Use a new `-rc.N` for a new
   candidate, a patch version for compatible fixes, and a minor version for new
   functionality in the current 0.x series. Never move an already published tag.
2. Update `CHANGELOG.md` and write `docs/releases/v<version>.md` with compatibility,
   validation, and checkpoint availability. Update the README's current release.
3. Run the relevant regression tests in the supported environment and commit
   the changes to `main`. Include full-model equivariance and optimizer checks
   when changing those implementations.
4. Tag that reviewed, clean commit and publish the notes. For example, after
   updating the files for the next candidate:

```bash
# Run from the repository root, on the reviewed release commit.
git status --short
python3 src/qhflow2_version.py
git tag -a v0.2.0-rc.2 -m 'QHFlow2 0.2.0-rc.2'
git push --atomic origin HEAD:main refs/tags/v0.2.0-rc.2
gh release create v0.2.0-rc.2 --verify-tag --prerelease \
  --title 'QHFlow2 v0.2.0-rc.2' \
  --notes-file docs/releases/v0.2.0-rc.2.md
```

GitHub release publication follows the [official release workflow](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
The commands above publish code; they do not launch training or publish weights.
The planned `v0.2.0` benchmark release follows Muon retraining and evaluation of
the eight QHFlow2/QHFlow2-O(3) models. Release candidates do not imply benchmark
accuracy. Include each published checkpoint's source tag, exact model/run config,
dataset split, validation results, and applicable metadata sidecars.
