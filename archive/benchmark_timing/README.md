# Historical training-time audit (excluded from benchmark)

The maintainer requested on 2026-10-21 that reconstructed original QHFlow2
training times be omitted. These files preserve the raw W&B timing history,
checkpoint/run associations and the attempted calculation for later review.
They are **not accepted training-cost measurements** and must not populate
the benchmark table, README, or checkpoint-release metadata.

The calculation used retained first/last event spans and discarded overwritten
resume tails. It excludes startup, restart gaps and other unobserved intervals;
later evaluations overwrote the W&B hardware metadata. Agreement with the
actual original training allocation has not been established.

The image and benchmark CSV intentionally leave original training GPU hours
blank. Updated-model training cost evidence is separate and remains displayed.
