import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOP_LEVEL = {
    "ASR",
    "Samples",
    "logs",
    "outputs",
    "wandb",
    "exp",
    "cross_val",
    "save_dir",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".ipynb", ".pdf", ".pkl", ".pt", ".pth"}


def publication_paths():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode()
    return [Path(item) for item in output.split("\0") if item]


def test_no_private_or_generated_artifacts_are_published():
    paths = publication_paths()
    assert not [path for path in paths if path.parts[0] in FORBIDDEN_TOP_LEVEL]
    assert not [path for path in paths if path.parts[:2] == ("docs", "superpowers")]
    assert not [path for path in paths if path.suffix.lower() in FORBIDDEN_SUFFIXES]


def test_no_machine_specific_paths_are_published():
    offenders = []
    for relative in publication_paths():
        if relative.parts[0] == "tests":
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
    for relative in publication_paths():
        if relative.parts[0] in {"Figure", "PROJECT_CONTEXT", "tests", "docs"}:
            continue
        if relative.suffix.lower() not in {".py", ".yaml", ".yml"}:
            continue
        text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore").lower()
        if "lipvoicer" in text:
            offenders.append(str(relative))
    assert offenders == []


def test_preprocessing_runtime_dependency_is_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "pathos" in requirements
