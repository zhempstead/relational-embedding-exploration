from gensim.models import KeyedVectors
from omegaconf import OmegaConf
import pandas as pd

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import make_symlink


def word2vec_model2emb(indir, outdir, cfg, table_csv):
    model_fname = "model"
    if cfg.model_suffix is not None:
        model_fname += f"_{cfg.model_suffix}"
    model = KeyedVectors.load_word2vec_format(indir / model_fname)
    model_cnf = OmegaConf.load(indir / "model_cnf")
    outfile = outdir / "embeddings.csv"

    word_dict = TokenDict()
    word_dict.load(indir / "word_dict.feather")

    df = pd.read_csv(table_csv)
    num_rows = len(df)

    if cfg.use_value_nodes:
        if model_cnf.rows:
            emb_df = get_base_row_val_vectors(model, word_dict, df)
        else:
            emb_df = get_base_val_vectors(model, word_dict, df)
    else:
        if not model_cnf.rows:
            raise ValueError("Must have row tokens if not using value nodes")
        emb_df = get_base_row_vectors(model, word_dict, num_rows)
    emb_df.to_csv(outfile, index=False)


def get_base_row_vectors(model, word_dict, num_rows):
    row_tokens = [f'base_row:{idx}' for idx in range(num_rows)]
    return pd.DataFrame([model[word_dict.getNumForToken(rt)] for rt in row_tokens])

def get_base_row_val_vectors(model, word_dict, df):
    df['base_row'] = [f'base_row:{idx}' for idx in range(len(df))]
    return get_base_val_vectors(model, word_dict, df)

def get_base_val_vectors(model, word_dict, df):
    for col in df.columns:
        df[col] = [word_dict.getNumForToken(rt) for rt in df[col]]
    return pd.DataFrame([sum(model[row[1:]]) for row in df.itertuples()])
