from pathlib import Path

from relational_embeddings.lib.utils import prev_stage_dir
from relational_embeddings.pipeline.model2emb.word2vec import word2vec_model2emb


def model2emb(cfg, outdir, indir=None):
    """
    Use model to produce embeddings on training and test data
    """
    if indir is None:
        indir = outdir.parent

    normalize_dir = prev_stage_dir(outdir, 'normalize')
    base_csv = normalize_dir / 'base.csv'

    print(f"Producing embeddings using '{cfg.model2emb.method}' method...")

    if cfg.model2emb.method == "word2vec":
        word2vec_model2emb(indir, outdir, cfg.model2emb, base_csv)
    else:
        raise ValueError("Unrecognized model2emb method '{cfg.model2emb.method}'")

    print(f"Done writing embeddings! Output at '{outdir}'")
