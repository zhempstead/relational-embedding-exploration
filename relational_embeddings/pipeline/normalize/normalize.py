from pathlib import Path

from relational_embeddings.pipeline.normalize.leva import leva_normalize


def normalize(cfg, outdir, indir=None):
    """
    Normalize input tables
    """
    if indir is None:
        indir = outdir.parent

    print(f"Normalizing using '{cfg.normalize.method}' method...")

    if cfg.normalize.method == "leva":
        leva_normalize(indir, outdir, cfg.normalize)
    else:
        raise ValueError("Unrecognized normalize method '{cfg.normalize.method}'")

    print(f"Done normalizing! Output at '{outdir}'")
