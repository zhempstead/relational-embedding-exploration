from pathlib import Path

from relational_embeddings.pipeline.text2model.word2vec import word2vec_text2model


def text2model(cfg, outdir, indir=None):
    """
    Convert normalized input tables to graph
    """
    if indir is None:
        indir = outdir.parent

    print(f"Converting graph to text using '{cfg.text2model.method}' method...")

    if cfg.text2model.method == "word2vec":
        word2vec_text2model(indir, outdir, cfg.text2model)
    else:
        raise ValueError("Unrecognized text2model method '{cfg.text2model.method}'")

    print(f"Done writing embeddings! Output at '{outdir}'")
