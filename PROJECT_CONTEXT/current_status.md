# Current status

## Public release scope

- Canonical WYS training and inference entry points are present.
- LRS2 and LRS3 are selected explicitly with `dataset=lrs2` or `dataset=lrs3`.
- The legacy classifier-guidance ASR subsystem, comparison samples, weights,
  datasets, logs, notebooks, and generated outputs are excluded.
- Website files and architecture figures are retained.

## Validation

The release is checked with `pytest -v`, Python `compileall`, `git diff
--check`, and a tracked-tree audit. The tests cover Hydra composition for both
datasets, dataset name normalization, canonical entry-point syntax and naming,
machine-local path rejection, and forbidden artifact rejection.

## Known limitations

- Full training and GPU inference have not been run as part of publication.
- WYS, HiFi-GAN, and single-video lip-reading weights are not distributed.
- LRS2/LRS3 data must be obtained and preprocessed by the user under the
  datasets' terms.
- Legacy dependency pins reflect the original research stack; the added WYS
  text-model and tooling packages are unpinned and may require a compatible
  CUDA-specific PyTorch installation.
