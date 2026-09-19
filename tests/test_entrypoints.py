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


def test_public_media_commands_do_not_invoke_a_shell():
    paths = ENTRYPOINTS + [
        ROOT / "dataloaders/extract_audio_from_video.py",
        ROOT / "mouthroi_processing/crop_and_infer.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "shell=True" not in source, path.name
        assert "os.system(" not in source, path.name


def test_real_video_frame_selection_stays_in_bounds():
    source = (ROOT / "inference_real_video.py").read_text(encoding="utf-8")
    assert "random.randint(0, face_crop_160.shape[0])" not in source
    assert "random.randrange(face_crop_160.shape[0])" in source


def test_full_split_writes_actual_ground_truth_audio():
    source = (ROOT / "inference_full_test_split.py").read_text(encoding="utf-8")
    assert "ground_truth_audio = gt_audio.squeeze().cpu().numpy()" in source
    assert "video_id + '_gt.wav'), ground_truth_audio" in source


def test_mouth_crop_helper_uses_package_imports():
    source = (ROOT / "dataloaders/extract_moutcrops.py").read_text(encoding="utf-8")
    assert "from pipelines." not in source
    assert "from mouthroi_processing.pipelines." in source
