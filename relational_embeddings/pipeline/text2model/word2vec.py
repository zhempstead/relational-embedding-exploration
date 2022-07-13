'''
Reference implementation of node2vec.
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''
from gensim.models import Word2Vec

from omegaconf import OmegaConf

def word2vec_text2model(indir, outdir, cfg):
  infile = indir / 'text.txt'
  outfile = outdir / 'embeddings'

  with open(infile) as f:
    text = [line.strip().split(' ') for line in f.readlines()]
  model = Word2Vec(text, size=cfg.dimensions, window=cfg.window_size, min_count=0, sg=1, workers=cfg.workers, iter=cfg.iter)
  model.wv.save_word2vec_format(outfile)

  (outdir / 'word_dict').symlink_to(indir / 'word_dict')

  model_cnf = OmegaConf.create({'model_type': 'word2vec'})
  word_types = OmegaConf.load(indir / 'word_types')
  model_cnf = OmegaConf.merge(model_cnf, word_types)
  OmegaConf.save(model_cnf, outdir / 'model_cnf')
