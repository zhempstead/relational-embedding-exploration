from pathlib import Path

from omegaconf import OmegaConf

from relational_embeddings.pipeline.table2graph.leva import leva_table2graph
from relational_embeddings.pipeline.table2graph.embdi import embdi_table2graph


def table2graph(cfg, outdir, indir=None):
    """
    Convert normalized input tables to graph
    """
    if indir is None:
        indir = outdir.parent

    print(f"Converting to graph using '{cfg.table2graph.method}' method...")

    if cfg.table2graph.method == "leva":
        leva_table2graph(indir, outdir, cfg.table2graph)
        node_types = {"values": True, "columns": False, "rows": True}
    elif cfg.table2graph.method == "embdi":
        embdi_table2graph(indir, outdir, cfg.table2graph)
        node_types = {"values": True, "columns": True, "rows": True}
    else:
        raise ValueError("Unrecognized table2graph method '{cfg.table2graph.method}'")

    node_types = OmegaConf.create(node_types)
    OmegaConf.save(node_types, outdir / "node_types")

    print(f"Done converting to graph! Output at '{outdir}'")
