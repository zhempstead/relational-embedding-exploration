"""
Reference implementation of node2vec.
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
"""
import time

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from omegaconf import OmegaConf

from relational_embeddings.lib.utils import make_symlink


def word2vec_text2model(indir, outdir, cfg):
    infile = indir / "text.txt"
    outfile = outdir / "model"
    outdir_tag = "dataset" + str(outdir).split('dataset')[1]

    model = Word2Vec(
        corpus_file=str(infile),
        vector_size=cfg.dimensions,
        window=cfg.window_size,
        min_count=0,
        sg=int(cfg.skipgram),
        hs=int(cfg.hierarchical_softmax),
        workers=int(cfg.workers),
        seed=int(cfg.random_seed),
        epochs=cfg.epochs,
    callbacks=[callback(outdir_tag)],
    )
    model.wv.save_word2vec_format(outfile)

    make_symlink(indir / "word_dict.feather", outdir / "word_dict.feather")

    model_cnf = OmegaConf.create({"model_type": "word2vec"})
    word_types = OmegaConf.load(indir / "word_types")
    model_cnf = OmegaConf.merge(model_cnf, word_types)
    OmegaConf.save(model_cnf, outdir / "model_cnf")

class callback(CallbackAny2Vec):
    '''Callback to print status after each epoch.'''

    def __init__(self, outdir):
        self.outdir = outdir
        self.epoch = 0
        self.start = time.perf_counter()

    def on_epoch_end(self, model):
        end = time.perf_counter()
        elapsed = round(end - self.start)
        print(f"{self.outdir}: epoch {self.epoch} done in {elapsed} seconds")
        self.epoch += 1
        self.start = end
