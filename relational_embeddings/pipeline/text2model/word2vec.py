"""
Reference implementation of node2vec.
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
"""
from gensim.models import Word2Vec

from omegaconf import OmegaConf

from relational_embeddings.lib.utils import make_symlink


def word2vec_text2model(indir, outdir, cfg):
    infile = indir / "text.txt"
    outfile = outdir / "model"

    model = Word2Vec(
        corpus_file=str(infile),
        size=cfg.dimensions,
        window=cfg.window_size,
        min_count=0,
        sg=int(cfg.skipgram),
        hs=int(cfg.hierarchical_softmax),
        workers=1,
        seed=int(cfg.random_seed),
        iter=cfg.iter,
    )
    model.wv.save_word2vec_format(outfile)

    make_symlink(indir / "word_dict.feather", outdir / "word_dict.feather")

    model_cnf = OmegaConf.create({"model_type": "word2vec"})
    word_types = OmegaConf.load(indir / "word_types")
    model_cnf = OmegaConf.merge(model_cnf, word_types)
    OmegaConf.save(model_cnf, outdir / "model_cnf")
