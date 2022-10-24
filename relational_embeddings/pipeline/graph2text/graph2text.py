from pathlib import Path

from relational_embeddings.lib.utils import make_symlink
from relational_embeddings.pipeline.graph2text.node2vec import node2vec_graph2text


def graph2text(cfg, outdir, indir=None):
    """
    Convert normalized input tables to graph
    """
    if indir is None:
        indir = outdir.parent

    print(f"Converting graph to text using '{cfg.graph2text.method}' method...")

    if cfg.graph2text.method == "node2vec":
        node2vec_graph2text(indir, outdir, cfg.graph2text)
    else:
        raise ValueError("Unrecognized graph2text method '{cfg.graph2text.method}'")

    make_symlink(indir / "node_dict.feather", outdir / "word_dict.feather")
    make_symlink(indir / "node_types", outdir / "word_types")

    print(f"Done converting to text! Output at '{outdir}'")
