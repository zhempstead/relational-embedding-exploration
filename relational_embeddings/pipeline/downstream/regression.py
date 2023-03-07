from relational_embeddings.lib.eval_utils import train_downstream_model
from relational_embeddings.lib.utils import prev_stage_dir

def regression_downstream(df_x, df_y, outdir, cfg):
    df_x = pd.read_csv(prev_stage_dir(outdir, "model2emb") / "embeddings.csv")
    df_y = pd.read_csv(prev_stage_dir(outdir, "dataset") / "base.csv")[[cfg.dataset.target_column]]
    teefile = outdir / 'results.txt'
    return train_downstream_model(df_x, df_y, cfg, outdir, teefile)
