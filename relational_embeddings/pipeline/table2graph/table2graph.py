from pathlib import Path

from relational_embeddings.pipeline.table2graph.leva import leva_table2graph

def table2graph(cfg, outdir, indir=None):
    '''
    Convert normalized input tables to graph
    '''
    if indir is None:
      indir = outdir.parent

    print(f"Converting to graph using '{cfg.table2graph.method}' method...")

    if cfg.table2graph.method == 'leva':
      leva_table2graph(indir, outdir, cfg.table2graph)
    else:
      raise ValueError("Unrecognized table2graph method '{cfg.table2graph.method}'")
    
    print(f"Done converting to graph! Output at '{outdir}'")

if __name__ == '__main__':
    table2graph()
