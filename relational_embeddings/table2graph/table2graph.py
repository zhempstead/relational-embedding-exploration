from pathlib import Path

import hydra

from relational_embeddings.table2graph.leva import leva_table2graph

@hydra.main(version_base=None, config_path='../../hydra_conf', config_name='run')
def table2graph(cfg):
    '''
    Convert normalized input tables to graph
    '''
    indir = Path.cwd() / 'normalize'
    outdir = Path.cwd() / 'graph'
    outdir.mkdir()

    print(f"Converting to graph using '{cfg.table2graph.method}' method...")

    if cfg.table2graph.method == 'leva':
      leva_table2graph(indir, outdir, cfg.table2graph)
    else:
      raise ValueError("Unrecognized table2graph method '{cfg.table2graph.method}'")
    
    print(f"Done converting to graph! Output at '{outdir}'")

if __name__ == '__main__':
    table2graph()
