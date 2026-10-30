# QH9Stable benchmark measurement notes

Results updated **2026-10-28** · updated models **v0.2.0-rc.2**.
See the [benchmark image](../../README.md#qh9stable-benchmark-update) and
[checkpoint downloads](../releases/v0.2.0-rc.2-checkpoints.md).

## Accuracy and model changes

Original-paper metrics, including QHNetV2, reproduce the supplied paper tables
at their reported precision. QHFlow2 parameter counts follow the
[checkpoint/config audit](../MODEL_CONFIGS.md); updated counts are measured
trainable parameters from the evaluated checkpoints.

Updated results use seed 42 and all **13,084** official QH9Stable random-split
test molecules. S, O(3)-S, M, L, and XL each completed 260,000 optimizer steps.
Their exported weights and reported results use the best validation checkpoint
at **step 258,409** (epoch 78).

Each updated model has paired **3-step then 1-step** accuracy rows from the
same recorded full-test evaluation and validation-selected checkpoint. Model,
parameter count, and training GPU hours appear once per pair. Original-paper
rows retain their reported metrics without assigning a new step label.

H, occupied orbital energy, HOMO, LUMO, and GAP errors are in **μEh**; Sc is
in **percent**. Original E/F values retain their source units. Updated E/F,
updated SCF metrics, and other unavailable measurements are shown as dashes;
no new full-test energy/force evaluation is claimed.

The updated models combine the order fix, **gate change (spectral)**, and Muon
at **1e-3**. Norm, embedding, and other unsupported parameters use AdamW at
**1e-4**. Updated S/M also enable the Hamiltonian-input path. These results
combine architecture and training changes; they are not a single-change ablation.

**+ O(3) group averaging** denotes the separately trained QHFlow2-O(3)-s model,
which averages shared-weight identity and inversion predictions. Both rows are
grey in the image. See [model and optimizer details](../QH9STABLE_MUON.md).

Updated H/orbital results are rounded to three decimals; SCF ratio and ERIC use
two decimals. More digits expose recorded differences and do not imply
statistical significance. The [unrounded results](../../reproduce/benchmark/current-results.json)
retain exact source metric keys for both step counts.

## Training cost

GPU hours are reported only for the completed updated models. They sum observed
spans in the retained resume lineage, including validation and checkpointing
within those spans, and excluding restart downtime.

S, O(3)-S, and M used one GPU; L used two; XL used four. XL completed with
**649.8 observed GPU hours**. Original-model training GPU hours are omitted.
See the [training-cost evidence](qh9-stable-training-costs-20261021.json).

## Inference speed

The image's single **Inference (samples/s)** column corresponds to the step
count on each updated row. Both step counts were measured independently for
all nine original/updated checkpoints; the
[full speed table and reproduction commands](../../reproduce/README.md#inference-speed)
include original-model speeds as well.

| Setting | Value |
| --- | --- |
| Hardware | One exclusive H100 80GB |
| Precision | FP32, high matmul precision |
| Execution | Eager, without `torch.compile` |
| Batch size | 8 |
| Molecules | Same fixed 128-molecule official test subset, same order |
| Repeats | One complete warmup sweep, then three timed sweeps per step count |
| Rate | 128 / median total timed seconds per sweep |
| Measurement dates | 2026-10-21 UTC for the first eight models; 2026-10-28 UTC for updated XL |

Timing includes actual flow sampling: the prior, geometry, and Hamiltonian-block
prediction. It excludes data loading, device transfer, batch cloning,
full-matrix assembly, accuracy metrics, and DFT. Each O(3) molecule counts once,
with both identity/inversion evaluations included in the timed work.

These are batched model-inference rates, not training throughput, end-to-end DFT
rates, or single-molecule latency. Training samples/s remains only in historical
evidence. The fixed batch size supports comparison and does not claim maximum
throughput for each model.

## SCF ratio and ERIC

Let `N_model` and `N_MinAO` be the paired SCF cycle counts for each molecule.
Both columns are percentages, with lower values better:

- **SCF ratio (%)** = `100 × sum(N_model) / sum(N_MinAO)`. MinAO is 100%.
- **ERIC (%)** = `100 × mean((N_model + 1) / N_MinAO)`. The extra one accounts
  for an initial Hamiltonian build under the
  [ERIC accounting convention](https://arxiv.org/html/2604.21657v1#S4).
  This is not a wall-time speedup.

The six available original-model entries are recomputed from
[300-molecule paired records](../../reproduce/scf/recorded-300-cycles.csv).
These are the same molecules used in
[original-paper Figure 4](https://arxiv.org/html/2602.16897v1#S4.F4), confirmed
by the maintainer on 2026-10-21. The
[molecule list](../../reproduce/scf/recorded-300.csv) links each sample to its
original QH9 index and molecular formula.

The [provenance record](../../reproduce/scf/provenance.json) preserves the
recovered March 2026 solver settings, including differences from Appendix E.4,
the cohort confirmation, and geometry matches. ERIC is a new post-hoc
calculation; these added columns are not transcribed paper-table values.
Updated-model SCF metrics have not been evaluated and remain dashes.

See the [recomputed summary](../../reproduce/scf/recorded-300-summary.json) and
[SCF / DFT evaluation guide](../../reproduce/README.md#scf-and-dft-energyforces).
The DFT workflow records functional, basis, overlap policy, grid/gradient
options, input/checkpoint hashes, package versions, per-molecule results,
and failures.
