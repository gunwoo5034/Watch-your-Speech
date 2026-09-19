# WYS Code Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a minimal, reproducible WYS training and inference codebase with explicit LRS2/LRS3 selection in `gunwoo5034/Watch-your-Speech`.

**Architecture:** Keep the website and architecture figures, remove comparison samples, and import only the WYS runtime from the research checkout. Hydra config groups own dataset selection, `WYSDataset` consumes an explicit dataset name, and canonical entry points use the text-conditioned WYS model without legacy ASR classifier guidance.

**Tech Stack:** Python 3.9, PyTorch, Hydra/OmegaConf, Transformers/PEFT, OpenCV, librosa, HiFi-GAN, pytest

**Spec:** `PROJECT_CONTEXT/publication_design.md`

## Global Constraints

- Preserve `index.html`, `index.css`, and `Figure/`.
- Remove `Samples/`; do not publish comparison media.
- Do not merge the unrelated local LipVoicer Git history.
- Exclude checkpoints, generated media, logs, datasets, notebooks, PDFs, ablations, metric outputs, and machine-specific paths.
- Public dataset selectors are `dataset=lrs2` and `dataset=lrs3`.
- Runtime selection reads `dataset.name`; it never guesses from path substrings.
- WYS modules use WYS naming while retaining explicit upstream attribution.
- Full training and GPU inference remain unverified unless actually run.

## Review Focus

- Mixed-case names (`LRS2`, `LRS3`) normalize correctly; Task 2 tests them.
- Unsupported names raise `ValueError` listing both choices; Task 2 tests this.
- Dataset roots with spaces do not affect selection; Task 2 tests this.
- Public inference never imports or initializes removed `ASR`; Task 3 audits its syntax and source.
- Generated artifacts and local paths cannot enter Git; Task 4 audits every tracked path and text file.

---

### Task 1: Establish the curated public tree

**Files:**
- Create: `.gitignore`, `LICENSE`, `requirements.txt`, `distributed_util.py`, `utils.py`
- Create: `train_melgen.py`, `inference_melgen.py`, `inference_full_test_split.py`, `inference_real_video.py`
- Create: `dataloaders/**`, `models/**`, `hifi_gan/{config.json,env.py,generator.py,utils.py}`, `mouthroi_processing/**`
- Delete: `Samples/**`
- Test: `tests/test_public_tree.py`

**Interfaces:**
- Consumes: `/home/gunwoo/WYS_RFM` as the read-only research source.
- Produces: A WYS source tree with canonical root entry points and no comparison media.

- [ ] **Step 1: Write the failing public-tree tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_core_entrypoints_are_published():
    required = {
        "train_melgen.py", "inference_melgen.py",
        "inference_full_test_split.py", "inference_real_video.py",
        "models/audiovisual_model.py", "models/transformer.py",
        "dataloaders/dataset_lipvoicer.py", "hifi_gan/generator.py",
        "mouthroi_processing/crop_and_infer.py",
    }
    assert not {path for path in required if not (ROOT / path).is_file()}

def test_comparison_samples_are_not_published():
    assert not (ROOT / "Samples").exists()

def test_website_and_figures_are_preserved():
    assert (ROOT / "index.html").is_file()
    assert (ROOT / "index.css").is_file()
    assert (ROOT / "Figure/architecture_1.png").is_file()
```

- [ ] **Step 2: Confirm the tests fail for the missing code and existing samples**

Run: `pytest tests/test_public_tree.py -v`

Expected: the first two tests fail for the intended reasons.

- [ ] **Step 3: Import the explicit core set and remove samples**

```bash
WYS_SOURCE_REPO=/home/gunwoo/WYS_RFM
git rm -r Samples
for path in LICENSE distributed_util.py utils.py train_melgen.py inference_melgen.py inference_real_video.py requirements.txt; do
  install -D -m 0644 "$WYS_SOURCE_REPO/$path" "$path"
done
install -D -m 0644 "$WYS_SOURCE_REPO/inference_full_test_split_1.py" inference_full_test_split.py
for directory in dataloaders models hifi_gan mouthroi_processing; do
  git -C "$WYS_SOURCE_REPO" ls-files "$directory" | while read -r path; do
    install -D -m 0644 "$WYS_SOURCE_REPO/$path" "$path"
  done
done
install -D -m 0644 "$WYS_SOURCE_REPO/models/transformer.py" models/transformer.py
install -D -m 0644 "$WYS_SOURCE_REPO/models/ema.py" models/ema.py
```

Do not copy `ASR/`, `hifi_gan/g_02400000`, benchmark weights, configs, notebooks, logs, or experiment outputs. Replace `.gitignore` with project entries for Python caches, environments, datasets, logs, WandB, `exp/`, checkpoints, `save_dir*/`, generated media, model weights, and notebooks.

- [ ] **Step 4: Verify and commit the curated tree**

```bash
pytest tests/test_public_tree.py -v
git add .gitignore LICENSE requirements.txt distributed_util.py utils.py train_melgen.py inference_melgen.py inference_full_test_split.py inference_real_video.py dataloaders models hifi_gan mouthroi_processing tests
git diff --cached --check
git diff --cached --stat
git commit -m "feat: publish curated WYS core"
```

Expected: 3 tests pass and no checkpoint/generated-media file is staged.

---

### Task 2: Add explicit LRS2/LRS3 config and WYS dataset naming

**Files:**
- Create: `configs/config.yaml`, `configs/dataset/lrs2.yaml`, `configs/dataset/lrs3.yaml`
- Create: `dataloaders/dataset_config.py`
- Rename: `dataloaders/dataset_lipvoicer.py` to `dataloaders/dataset_wys.py`
- Modify: `dataloaders/dataset_wys.py`, `dataloaders/__init__.py`
- Modify: `tests/test_public_tree.py`
- Test: `tests/test_dataset_config.py`

**Interfaces:**
- Consumes: Hydra `dataset=<choice>` overrides.
- Produces: `normalize_dataset_name(name: str) -> str`, `WYSDataset`, and explicit dataset config fields.

- [ ] **Step 1: Write failing normalization and composition tests**

```python
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from dataloaders.dataset_config import normalize_dataset_name

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

@pytest.mark.parametrize("raw,expected", [
    ("lrs2", "lrs2"), ("LRS2", "lrs2"),
    ("lrs3", "lrs3"), ("LRS3", "lrs3"),
])
def test_normalize_supported_dataset_names(raw, expected):
    assert normalize_dataset_name(raw) == expected

def test_reject_unsupported_dataset_name():
    with pytest.raises(ValueError, match="lrs2.*lrs3"):
        normalize_dataset_name("lrw")

@pytest.mark.parametrize("choice,name,root", [
    ("lrs2", "lrs2", "datasets/LRS2"),
    ("lrs3", "lrs3", "datasets/LRS3"),
])
def test_hydra_dataset_choice(choice, name, root):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"dataset={choice}"])
    assert cfg.dataset.name == name
    assert cfg.dataset.root == root
    assert cfg.dataset.videos_dir == f"{root}/videos"
    assert cfg.dataset.audios_dir == f"{root}/audios"
    assert cfg.dataset.mouthrois_dir == f"{root}/mouth_rois"

def test_dataset_choice_does_not_depend_on_root_name():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[
            "dataset=lrs2", "dataset.root=/data/a neutral folder"
        ])
    assert cfg.dataset.name == "lrs2"
```

- [ ] **Step 2: Confirm collection fails because config modules do not exist**

Run: `pytest tests/test_dataset_config.py -v`

Expected: missing `dataloaders.dataset_config` or dataset config group.

- [ ] **Step 3: Implement the name validator**

```python
SUPPORTED_DATASETS = ("lrs2", "lrs3")

def normalize_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Choose one of: {supported}")
    return normalized
```

- [ ] **Step 4: Create the Hydra config group**

Start `configs/config.yaml` with:

```yaml
defaults:
  - dataset: lrs3
  - _self_
```

Retain the research model/diffusion/text/attention/audio/train/generate values. Create `configs/dataset/lrs2.yaml` as:

```yaml
name: lrs2
root: datasets/LRS2
videos_dir: ${dataset.root}/videos
mouthrois_dir: ${dataset.root}/mouth_rois
audios_dir: ${dataset.root}/audios
text_dir: ${dataset.root}/videos
dataset_root: ${dataset.videos_dir}
sampling_rate: ${audio.sampling_rate}
videos_window_size: 25
audio_stft_hop: ${audio.hop_length}
pred_text_main_dir: null
pred_text_pretrain_dir: null
```

Create `configs/dataset/lrs3.yaml` with identical keys, `name: lrs3`, and `root: datasets/LRS3`.

- [ ] **Step 5: Rename the dataset and make selection explicit**

Run: `git mv dataloaders/dataset_lipvoicer.py dataloaders/dataset_wys.py`

Use this constructor interface and replace path-substring branches with the normalized name:

```python
from .dataset_config import normalize_dataset_name

class WYSDataset(torch.utils.data.Dataset):
    def __init__(self, split, name, root, videos_dir, mouthrois_dir, audios_dir,
                 text_dir, sampling_rate, videos_window_size, audio_stft_hop,
                 dataset_root, pred_text_main_dir=None, pred_text_pretrain_dir=None):
        self.dataset_name = normalize_dataset_name(name)
        self.ds_name = self.dataset_name.upper()
```

Use `if self.dataset_name == "lrs3"` and `elif self.dataset_name == "lrs2"`. Update `dataloaders/__init__.py` to import and instantiate `WYSDataset`. Change the Task 1 manifest assertion from `dataloaders/dataset_lipvoicer.py` to `dataloaders/dataset_wys.py` so the cumulative suite describes the final public API.

- [ ] **Step 6: Run focused tests and commit**

```bash
pytest tests/test_dataset_config.py tests/test_public_tree.py -v
git diff --check
git add configs dataloaders tests
git commit -m "feat: select LRS2 or LRS3 with Hydra"
```

Expected: all focused tests pass and `dataset_lipvoicer.py` no longer exists.

---

### Task 3: Make canonical entry points use WYS

**Files:**
- Modify: `train_melgen.py`, `inference_melgen.py`, `inference_full_test_split.py`, `inference_real_video.py`
- Modify: `dataloaders/wav2mel.py`, `models/model_builder.py`, `models/transformer.py`
- Modify: `models/lipreading_models/lipreading_model.py`, `mouthroi_processing/crop_and_infer.py`
- Test: `tests/test_entrypoints.py`

**Interfaces:**
- Consumes: `cfg.dataset.name`, `WYSDataset`, and the text-conditioned `AudioVisualModel`.
- Produces: canonical training, single-video inference, and dataset inference commands without legacy ASR initialization or hardcoded local paths.

- [ ] **Step 1: Write failing entry-point audits**

```python
import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [ROOT / name for name in (
    "train_melgen.py", "inference_melgen.py",
    "inference_full_test_split.py", "inference_real_video.py",
)]

@pytest.mark.parametrize("path", ENTRYPOINTS)
def test_entrypoint_is_valid_python(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

@pytest.mark.parametrize("path", ENTRYPOINTS)
def test_entrypoint_uses_wys_dataset_name(path):
    source = path.read_text(encoding="utf-8")
    assert "dataset_lipvoicer" not in source
    assert "LipVoicerDataset" not in source

@pytest.mark.parametrize("name", ["inference_full_test_split.py", "inference_real_video.py"])
def test_wys_inference_does_not_load_legacy_asr(name):
    source = (ROOT / name).read_text(encoding="utf-8")
    assert "import ASR" not in source
    assert "asr_models" not in source
    assert "get_models(" not in source

def test_public_python_has_no_local_home_paths():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path.parts[len(ROOT.parts)] in {"tests", "docs", "PROJECT_CONTEXT"}:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in source for token in ("/home/gunwoo", "/workspace/inputs", "/disk2/")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
```

- [ ] **Step 2: Confirm obsolete identifiers and paths fail**

Run: `pytest tests/test_entrypoints.py -v`

Expected: the naming, ASR, and local-path assertions fail on imported research files.

- [ ] **Step 3: Update all dataset imports**

Use this interface in retained callers:

```python
from dataloaders.dataset_wys import WYSDataset

dataset = WYSDataset("test", **dataset_cfg)
```

- [ ] **Step 4: Remove classifier-guidance-only ASR code**

Remove `ASR.asr_models`, `get_models`, and the `asr_guidance_net`, `w_asr`, `asr_start`, `tokenizer`, and `decoder` parameters from WYS inference. Keep `guidance_text` as the text passed to `net.sample`. Derive output labels from `dataset_cfg.name.upper()`. The sampler interface becomes:

```python
def sampling(net, seed, diffusion_hyperparams, w_video, condition, guidance_text):
    mouthroi, face_image = condition
    sample_step = diffusion_hyperparams["T"]
    timesteps = torch.linspace(1.0, 0.0, sample_step + 1)
    torch.manual_seed(seed)
    eps0 = torch.randn(mouthroi.shape[0], 80, mouthroi.shape[2] * 4, device="cuda")
    x = eps0.clone()
    with torch.no_grad():
        for index in range(sample_step):
            timestep = timesteps[index]
            dt = timestep - timesteps[index + 1]
            steps = timestep * torch.ones((x.shape[0], 1), device=x.device)
            conditional = net.sample(x, eps0, mouthroi, face_image, guidance_text, steps, 0)
            unconditional = net.sample(x, eps0, mouthroi, face_image, guidance_text, steps, 1)
            x = x - ((1 + w_video) * conditional - w_video * unconditional) * dt
    return x
```

- [ ] **Step 5: Remove machine-local paths and debug-only mains**

Resolve resources relative to the module:

```python
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
config_path = MODEL_DIR / "lrw_snv1x_tcn2x.json"
```

Remove standalone debug `main` blocks from `models/transformer.py` and the lip-reading model. Make `mouthroi_processing/crop_and_infer.py` accept its existing `video_filename` and `output_directory` arguments without embedding an example path.

- [ ] **Step 6: Align Hydra decorators and experiment branding**

Use `config_name="config"` in all public entry points and `dataloaders/wav2mel.py`. Replace WandB setup with:

```python
wandb.init(project="Watch-Your-Speech")
```

Default CUDA device selection is 0 or the supplied `cfg.rank`; no script hardcodes device 1. Do not change model hyperparameters or dataset split behavior.

- [ ] **Step 7: Verify and commit canonical entry points**

```bash
pytest tests/test_entrypoints.py tests/test_dataset_config.py -v
python -m compileall -q train_melgen.py inference_melgen.py inference_full_test_split.py inference_real_video.py dataloaders models hifi_gan mouthroi_processing
git diff --check
git add train_melgen.py inference_melgen.py inference_full_test_split.py inference_real_video.py dataloaders models mouthroi_processing tests
git commit -m "refactor: expose canonical WYS entry points"
```

Expected: focused tests pass and compilation exits 0.

---

### Task 4: Document and audit the public release

**Files:**
- Modify: `README.md`, `requirements.txt`, `.gitignore`, `PROJECT_CONTEXT/README.md`
- Create: `THIRD_PARTY_NOTICES.md`, `PROJECT_CONTEXT/current_status.md`
- Create: `tests/test_publication_audit.py`

**Interfaces:**
- Consumes: canonical commands/configs from Tasks 2–3.
- Produces: public documentation and a machine-checkable artifact/branding audit.

- [ ] **Step 1: Write the failing publication audit**

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOP_LEVEL = {"ASR", "Samples", "logs", "outputs", "wandb", "exp", "cross_val", "save_dir"}
FORBIDDEN_SUFFIXES = {".ckpt", ".ipynb", ".pdf", ".pkl", ".pt", ".pth"}

def tracked_paths():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
    return [Path(item) for item in output.split("\0") if item]

def test_no_private_or_generated_artifacts_are_tracked():
    paths = tracked_paths()
    assert not [path for path in paths if path.parts[0] in FORBIDDEN_TOP_LEVEL]
    assert not [path for path in paths if path.suffix.lower() in FORBIDDEN_SUFFIXES]

def test_no_machine_specific_paths_are_tracked():
    offenders = []
    for relative in tracked_paths():
        if relative.parts[0] in {"tests", "docs", "PROJECT_CONTEXT"}:
            continue
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".wav"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in ("/home/gunwoo", "/disk2/", "/workspace/inputs")):
            offenders.append(str(relative))
    assert offenders == []

def test_readme_documents_both_dataset_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "dataset=lrs2" in readme
    assert "dataset=lrs3" in readme
    assert "Watch Your Speech" in readme

def test_obsolete_identity_is_not_used_for_wys_modules():
    offenders = []
    for relative in tracked_paths():
        if relative.parts[0] in {"Figure", "PROJECT_CONTEXT", "tests", "docs"}:
            continue
        if relative.suffix.lower() not in {".py", ".yaml", ".yml"}:
            continue
        text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore").lower()
        if "lipvoicer" in text:
            offenders.append(str(relative))
    assert offenders == []
```

- [ ] **Step 2: Confirm README/artifact checks fail where expected**

Run: `pytest tests/test_publication_audit.py -v`

Expected: README dataset command coverage and any remaining path/identity checks fail.

- [ ] **Step 3: Rewrite README for WYS**

Document these exact public command forms:

```bash
python train_melgen.py dataset=lrs2
python train_melgen.py dataset=lrs3
python inference_full_test_split.py dataset=lrs2 generate.ckpt_path=/path/to/wys-lrs2.pkl generate.lipread_text_dir=/path/to/predicted_text
python inference_real_video.py dataset=lrs3 generate.ckpt_path=/path/to/wys-lrs3.pkl generate.video_path=/path/to/input.mp4
```

Include installation, data layouts, preprocessing, checkpoint exclusion, architecture, license, and acknowledgements. Do not invent checkpoint links or citation metadata.

- [ ] **Step 4: Reconcile dependencies and attribution**

Retain required research pins and add imports missing from the source requirements:

```text
peft
pytest>=8.1,<9
tensorboard
transformers
wandb
```

Create `THIRD_PARTY_NOTICES.md` naming DiffWave/Sashimi, LipVoicer, Visual Speech Recognition for Multiple Languages, and HiFi-GAN for retained source. Record test evidence and the unverified GPU/full-training limitation in `PROJECT_CONTEXT/current_status.md`.

- [ ] **Step 5: Run the full lightweight suite and commit**

```bash
pytest -v
python -m compileall -q train_melgen.py inference_melgen.py inference_full_test_split.py inference_real_video.py dataloaders models hifi_gan mouthroi_processing
git diff --check
git add README.md requirements.txt .gitignore THIRD_PARTY_NOTICES.md PROJECT_CONTEXT tests
git commit -m "docs: document reproducible WYS workflows"
```

Expected: pytest reports zero failures and compileall exits 0.

---

### Task 5: Verify and deliver to `main`

**Files:**
- Modify: `PROJECT_CONTEXT/current_status.md` only when final evidence differs from Task 4.

**Interfaces:**
- Consumes: all implementation commits and remote `publication`.
- Produces: verified target `main` with the curated WYS code.

- [ ] **Step 1: Run fresh final verification**

```bash
pytest -v
python -m compileall -q train_melgen.py inference_melgen.py inference_full_test_split.py inference_real_video.py dataloaders models hifi_gan mouthroi_processing
git diff --check
git status --short --branch
if git ls-files | rg '(^|/)(Samples|ASR|logs|outputs|wandb|exp|cross_val|save_dir[^/]*)/'; then exit 1; fi
if git grep -n -E '/home/gunwoo|/disk2/|/workspace/inputs' -- '*.py' '*.yaml' '*.yml' ':!tests/**' ':!docs/**'; then exit 1; fi
```

Expected: zero test failures, compileall exit 0, a clean worktree, and no audit matches.

- [ ] **Step 2: Inspect the exact publication delta**

```bash
git log --oneline publication/main..HEAD
git diff --stat publication/main...HEAD
git diff --name-status publication/main...HEAD
git count-objects -vH
```

Expected: website/figures remain, `Samples/` is removed, only curated code/docs/tests are added, and no large checkpoint object appears.

- [ ] **Step 3: Push the branch and fast-forward `main`**

```bash
git push publication publish/wys-code
git fetch publication main
git merge-base --is-ancestor publication/main HEAD
git push publication HEAD:main
```

Expected: `main` fast-forwards without force push.

- [ ] **Step 4: Verify the remote**

```bash
local_head=$(git rev-parse HEAD)
remote_head=$(git ls-remote publication refs/heads/main | awk '{print $1}')
test "$local_head" = "$remote_head"
gh repo view gunwoo5034/Watch-your-Speech --json defaultBranchRef,url
```

Expected: local HEAD equals remote `main`, whose default-branch metadata is returned by GitHub.
