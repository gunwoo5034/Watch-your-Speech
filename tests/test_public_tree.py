from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_entrypoints_are_published():
    required = {
        "train_melgen.py",
        "inference_melgen.py",
        "inference_full_test_split.py",
        "inference_real_video.py",
        "models/audiovisual_model.py",
        "models/transformer.py",
        "dataloaders/dataset_wys.py",
        "hifi_gan/generator.py",
        "mouthroi_processing/crop_and_infer.py",
    }
    assert not {path for path in required if not (ROOT / path).is_file()}


def test_comparison_samples_are_not_published():
    assert not (ROOT / "Samples").exists()


def test_website_and_figures_are_preserved():
    assert (ROOT / "index.html").is_file()
    assert (ROOT / "index.css").is_file()
    assert (ROOT / "Figure/architecture_1.png").is_file()
