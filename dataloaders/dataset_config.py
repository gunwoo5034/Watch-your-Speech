SUPPORTED_DATASETS = ("lrs2", "lrs3")


def normalize_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Choose one of: {supported}")
    return normalized
