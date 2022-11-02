import pandas as pd

from relational_embeddings.lib.utils import get_sweep_vars, prev_stage_dir
from relational_embeddings.pipeline.downstream.classification import classification_downstream
from relational_embeddings.pipeline.downstream.regression import regression_downstream

def downstream(cfg, outdir, indir=None):
    """
    Evaluate model on downstream ML task (as determined by dataset)
    """
    if indir is None:
        indir = outdir.parent

    teefile = outdir / 'results.txt'

    df_x = pd.read_csv(indir / 'embeddings.csv')
    df_y = pd.read_csv(prev_stage_dir(outdir, "dataset") / "base.csv")[[cfg.dataset.target_column]]

    if cfg.downstream.task == "classification":
        df = classification_downstream(df_x, df_y, outdir, cfg.downstream)
    elif cfg.downstream.task == "regression":
        df = regression_downstream(df_x, df_y, outdir, cfg.downstream)
    else:
        raise ValueError("Unrecognized downstream task '{cfg.downstream.task}'")

    orig_cols = list(df.columns)
    sweep_vars = get_sweep_vars(outdir)
    for var, val in sweep_vars.items():
        df[var] = val
    df = df[list(sweep_vars.keys()) + orig_cols]
    df.to_csv(outdir / 'results.csv', index=False)

    print(f"Done with {cfg.downstream.task}! Results at '{outdir}'")
