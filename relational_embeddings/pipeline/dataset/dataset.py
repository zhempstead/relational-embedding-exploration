from pathlib import Path

from hydra.utils import get_original_cwd

from relational_embeddings.pipeline.dataset.real import real_dataset


def dataset(cfg, outdir, indir=None):
    """
    Normalize input tables
    """
    assert indir is None, "dataset should be the first step in the pipeline"

    print(f"Generating or preprocessing dataset...")

    if cfg.dataset.method == "real":
        real_dataset(outdir, cfg.dataset)
    else:
        raise ValueError("Unrecognized dataset method '{cfg.dataset.method}'")

    print(f"Done normalizing! Output at '{outdir}'")
