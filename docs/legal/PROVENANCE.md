# Provenance audit

This file records the evidence used for the repository's file-level licensing.
It is an engineering audit, not a substitute for a legal review or confirmation
from every copyright holder.

## Audit snapshot

- QHFlow2 initial import: `82ebba73ff510f4b2a49f3fef7646fd3446efd2f`
- AIRS comparison commit: `4a16c68a7da707c521019067dec51c227c10de45`
- FairChem comparison commit: `ecc7718769368804ebb683c719405c23ab5d84b8`
- e3nn dependency/license reference: tag `0.5.1`
- Audit date: 2026-07-28

The QHFlow2 repository history begins with a single 273-file import, so commit
authorship alone cannot establish originality. The classifications below use
content comparison, existing source comments, upstream license files, and the
project README.

## High-confidence classifications

| QHFlow2 material | Evidence | Classification |
| --- | --- | --- |
| `src/models/Real_QHNet.py` | 56% Git rename similarity to AIRS/QHNet `ori_QHNet_with_bias.py` | GPL-3.0-only |
| `src/models/Real_QHNet_qh9.py` | 49% similarity to AIRS/QHNet `ori_QHNet_wo_bias.py` | GPL-3.0-only |
| `src/models/QHFlow.py` | 15% copy similarity to AIRS/QHNet plus shared QHNet architecture | GPL-3.0-only |
| QHNet baseline layers, utilities, and MD17 configs listed in `REUSE.toml` | Direct or substantial AIRS/QHNet ancestry | GPL-3.0-only |
| `qh9_dataset.py` and its compatibility entrypoints | Independent QHFlow2 implementation using the published SQLite schema and split protocol as interoperability facts | MIT |
| Legacy Rowan QH9 loader and QH9 dataset configs listed in `REUSE.toml` | AIRS/QH9 and QHNet software ancestry | GPL-3.0-only |
| QHFlow2 checkpoint weights trained on QH9 | Project release policy for learned QH9 artifacts | CC-BY-NC-SA-4.0 |
| `src/models/modules/escn_*.py` | Existing Meta MIT headers and direct comparison to FairChem | MIT |
| `src/models/modules/Jd.pt` | SHA-256 is byte-identical to FairChem UMA `Jd.pt` | MIT |
| `dataset/water/**` and `dataset/water_shard/**` | Raw archive is the published SchNOrb Hamiltonian water dataset | MIT |
| Remaining project-authored files | No substantial match found in the audited upstream paths | MIT, subject to author review |

The exact upstream MIT notices for the audited FairChem commit and the pinned
e3nn 0.5.1 dependency are included under
`docs/legal/third_party_licenses/`. e3nn API imports do not by themselves make
QHFlow2 source files e3nn-derived, so no QHFlow2 file was reclassified solely
because it imports e3nn.

The QHFlow2-specific additions to third-party-derived files are intended to be
permissively reusable where they can be separated, but the combined files
remain under the upstream license shown in `REUSE.toml`.

## Resolved audit items

- The direct AIRS/QH9-style loader implementations were replaced on 2026-07-28
  by `src/dataset_module/qh9_dataset.py` plus thin compatibility entrypoints.
  The replacement preserves QHFlow2's public data contract and is MIT-licensed.
- The duplicated attributed time-embedding snippets were replaced by the
  independently authored `src/models/time_embedding.py` implementation.

## Remaining release blocker

- `src/models/modules/__init__.py` and the radial-basis implementation trace
  through AIRS/QHNet to PhiSNet. The current mapping conservatively follows
  AIRS/QHNet's GPL-3.0-only distribution pending clearer upstream evidence.

## QH9 license scope

By maintainer direction, QHFlow2 applies CC-BY-NC-SA-4.0 only to QHFlow2
checkpoint weight files trained on QH9 Stable or QH9 Dynamic. It does not
apply that license to source code, configuration, documentation, or checkpoint
metadata. The native QH9 loader and compatibility entrypoints are original
QHFlow2 software mapped to MIT. Legacy QH9 material identified in `REUSE.toml`
remains GPL-3.0-only, as does AIRS/QHNet-derived model software; other original
QHFlow2 software is mapped to MIT.

The external QH9 dataset remains a separately licensed upstream work under
CC-BY-NC-SA-4.0. Recording that upstream dataset license in this document does
not extend CC-BY-NC-SA-4.0 to QHFlow2 source code.
