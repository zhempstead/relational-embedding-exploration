from pathlib import Path

from omegaconf import OmegaConf

from relational_embeddings.pipeline.table2text.naive import naive_table2text


def table2text(cfg, outdir, indir=None):
    """
    Convert normalized input tables to text
    """
    if indir is None:
        indir = outdir.parent

    print(f"Converting to text using '{cfg.table2text.method}' method...")

    if cfg.table2text.method == "naive":
        naive_table2text(indir, outdir, cfg.table2text)
        node_types = {"values": True, "columns": False, "rows": False}
    else:
        raise ValueError(f"Unrecognized table2text method '{cfg.table2text.method}'")

    node_types = OmegaConf.create(node_types)
    OmegaConf.save(node_types, outdir / "word_types")

    print(f"Done converting to text! Output at '{outdir}'")
