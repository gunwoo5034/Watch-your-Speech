import ast
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    ROOT / name
    for name in (
        "train_melgen.py",
        "inference_melgen.py",
        "inference_full_test_split.py",
        "inference_real_video.py",
    )
]


@pytest.mark.parametrize("path", ENTRYPOINTS)
def test_entrypoint_is_valid_python(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", ENTRYPOINTS)
def test_entrypoint_uses_wys_dataset_name(path):
    source = path.read_text(encoding="utf-8")
    assert "dataset_lipvoicer" not in source
    assert "LipVoicerDataset" not in source


@pytest.mark.parametrize(
    "name", ["inference_full_test_split.py", "inference_real_video.py"]
)
def test_wys_inference_does_not_load_legacy_asr(name):
    source = (ROOT / name).read_text(encoding="utf-8")
    assert "import ASR" not in source
    assert "asr_models" not in source
    assert "get_models(" not in source


def test_public_python_has_no_local_home_paths():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path.relative_to(ROOT).parts[0] in {"tests", "docs", "PROJECT_CONTEXT"}:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in source for token in ("/home/gunwoo", "/workspace/inputs", "/disk2/")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_mel_generation_main_passes_only_supported_generate_options():
    source = (ROOT / "inference_melgen.py").read_text(encoding="utf-8")
    assert "**cfg.generate" not in source
    assert "n_samples=cfg.generate.n_samples" in source
    assert "w_video=cfg.generate.w_video" in source
