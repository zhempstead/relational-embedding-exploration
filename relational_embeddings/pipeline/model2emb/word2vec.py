from gensim.models import KeyedVectors
from omegaconf import OmegaConf
import pandas as pd

from relational_embeddings.lib.token_dict import TokenDict
from relational_embeddings.lib.utils import make_symlink


def word2vec_model2emb(indir, outdir, cfg, table_csv):
    model = KeyedVectors.load_word2vec_format(indir / "model")
    model_cnf = OmegaConf.load(indir / "model_cnf")
    outfile = outdir / "embeddings"

    word_dict = TokenDict()
    word_dict.load(indir / "word_dict")

    df = pd.read_csv(table_csv)
    num_rows = len(df)

    emb_df = get_base_row_vectors(model, word_dict, num_rows)
    emb_df.to_csv(outfile, index=False)


def get_base_row_vectors(model, word_dict, num_rows):
    row_tokens = [f'base_row:{idx}' for idx in range(num_rows)]
    return pd.DataFrame([model[word_dict.getNumForToken(rt)] for rt in row_tokens])
