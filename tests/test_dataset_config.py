import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / "configs")


def load_dataset_config():
    module_path = ROOT / "dataloaders" / "dataset_config.py"
    spec = importlib.util.spec_from_file_location("dataset_config", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("lrs2", "lrs2"), ("LRS2", "lrs2"), ("lrs3", "lrs3"), ("LRS3", "lrs3")],
)
def test_normalize_supported_dataset_names(raw, expected):
    assert load_dataset_config().normalize_dataset_name(raw) == expected


def test_reject_unsupported_dataset_name():
    with pytest.raises(ValueError, match="lrs2.*lrs3"):
        load_dataset_config().normalize_dataset_name("lrw")


@pytest.mark.parametrize(
    ("choice", "name", "root"),
    [("lrs2", "lrs2", "datasets/LRS2"), ("lrs3", "lrs3", "datasets/LRS3")],
)
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
        cfg = compose(
            config_name="config",
            overrides=["dataset=lrs2", "dataset.root='/data/a neutral folder'"],
        )
    assert cfg.dataset.name == "lrs2"
