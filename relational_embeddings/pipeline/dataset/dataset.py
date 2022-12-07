from pathlib import Path

from hydra.utils import get_original_cwd

from relational_embeddings.pipeline.dataset.real import real_dataset
from relational_embeddings.pipeline.dataset.single_join import single_join_dataset
from relational_embeddings.pipeline.dataset.double_join import double_join_dataset
from relational_embeddings.pipeline.dataset.single_table import single_table_dataset


def dataset(cfg, outdir, indir=None):
    """
    Symlink real dataset or create synthetic dataset
    """
    assert indir is None, "dataset should be the first step in the pipeline"

    print(f"Generating or preprocessing dataset...")

    if cfg.dataset.method == "real":
        real_dataset(outdir, cfg.dataset)
    elif cfg.dataset.method == "single_join":
        single_join_dataset(outdir, cfg.dataset)
    elif cfg.dataset.method == "double_join":
        double_join_dataset(outdir, cfg.dataset)
    elif cfg.dataset.method == "single_table":
        single_table_dataset(outdir, cfg.dataset)
    else:
        raise ValueError(f"Unrecognized dataset method '{cfg.dataset.method}'")

    print(f"Done with dataset! Output at '{outdir}'")
