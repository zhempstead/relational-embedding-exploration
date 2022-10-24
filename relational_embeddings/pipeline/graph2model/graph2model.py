from pathlib import Path

from relational_embeddings.lib.utils import make_symlink
from relational_embeddings.pipeline.graph2model.proNE import proNE_graph2model


def graph2model(cfg, outdir, indir=None):
    """
    Convert normalized input tables to graph
    """
    if indir is None:
        indir = outdir.parent

    print(f"Converting graph to model using '{cfg.graph2model.method}' method...")

    if cfg.graph2model.method == "ProNE":
        proNE_graph2model(indir, outdir, cfg.graph2model)
    else:
        raise ValueError("Unrecognized graph2model method '{cfg.graph2model.method}'")

    make_symlink(indir / "node_dict.feather", outdir / "word_dict.feather")
    make_symlink(indir / "node_types", outdir / "word_types")

    print(f"Done creating model! Output at '{outdir}'")
