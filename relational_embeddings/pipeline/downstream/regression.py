from relational_embeddings.lib.eval_utils import train_downstream_model

def regression_downstream(df_x, df_y, outdir, cfg):
    teefile = outdir / 'results.txt'
    return train_downstream_model(df_x, df_y, cfg, outdir, teefile)
