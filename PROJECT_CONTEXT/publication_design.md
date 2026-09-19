# WYS code publication design

## Goal

Publish the reproducible core implementation of **Watch Your Speech (WYS)** in
`gunwoo5034/Watch-your-Speech` without exposing local experiment artifacts or
retaining LipVoicer as the identity of the implementation.

The existing project website and architecture figures remain available. The
existing comparison samples are removed from the repository.

## Repository strategy

- Use the existing `Watch-your-Speech` repository and its history as the base.
- Add the curated WYS implementation to the repository root.
- Do not merge the unrelated local LipVoicer Git history.
- Preserve `index.html`, `index.css`, and `Figure/`.
- Remove `Samples/` from the public repository.

## Public code scope

Include only code required for:

- WYS model construction and text-video fusion;
- training;
- inference on one video;
- inference on an LRS2 or LRS3 test split;
- LRS2/LRS3 data preparation and loading;
- mouth-ROI and lip-reading preprocessing used by inference;
- HiFi-GAN source/configuration required for vocoding; and
- lightweight configuration and validation tests.

Exclude:

- checkpoints, generated audio/video, spectrograms, logs, and WandB state;
- local datasets and machine-specific paths;
- notebooks and exploratory plots;
- rebuttal-only baselines, metric scripts, ablations, and scratch configs;
- internal review PDFs and working notes;
- duplicate or superseded training/inference entry points;
- the legacy ASR classifier-guidance subsystem, which is not part of the WYS
  text-conditioning inference path; and
- comparison samples, including the existing `Samples/` directory.

## Dataset interface

Hydra config groups provide the public dataset selector:

```bash
python train_melgen.py dataset=lrs2
python train_melgen.py dataset=lrs3
python inference_full_test_split.py dataset=lrs2
python inference_full_test_split.py dataset=lrs3
```

`configs/dataset/lrs2.yaml` and `configs/dataset/lrs3.yaml` define the dataset
name, default relative root, split layout, and derived video/audio/mouth-ROI
paths. Any path remains overridable from the command line. No user- or
machine-specific absolute path is committed.

Dataset selection is explicit. Runtime code reads `dataset.name`; it does not
guess the dataset from substrings in a filesystem path. Unsupported names fail
with a clear error.

## Naming and attribution

- Rename `dataloaders/dataset_lipvoicer.py` to `dataloaders/dataset_wys.py`.
- Rename `LipVoicerDataset` to `WYSDataset` and update all public imports.
- Use `Watch-Your-Speech` for experiment/logging project names.
- Rewrite README commands, descriptions, links, and headings for WYS.
- Retain legally and scientifically relevant attribution to upstream projects
  in source headers and an README acknowledgements section.
- Mentions of LipVoicer are permitted only when identifying the cited upstream
  work or a scientific baseline; they must not describe WYS itself.

## Documentation

The README documents:

- WYS purpose and architecture at a high level;
- supported LRS2/LRS3 selection commands;
- expected data directory layouts;
- installation and preprocessing;
- training and inference commands;
- checkpoint requirements without committing model weights; and
- upstream attribution and the WYS citation when citation metadata is known.

Unknown checkpoint URLs or publication metadata are not fabricated. They are
documented as unavailable when no verified public location exists.

## Validation

Automated tests cover:

- Hydra composition with both `dataset=lrs2` and `dataset=lrs3`;
- the selected dataset name and derived relative paths;
- rejection of unsupported dataset names where runtime validation occurs; and
- absence of obsolete LipVoicer implementation identifiers from public WYS
  modules.

Additional checks include Python compilation, configuration loading, README
command review, ignored-artifact verification, and a final public tree-size
inspection. Full training and GPU inference are out of scope for publication
validation and will be reported as unverified.

## Delivery

Implementation is committed in reviewable units on `publish/wys-code`, then
merged into and pushed to the target repository's `main` branch only after the
tests and publication audit pass. The local research worktree remains intact.
