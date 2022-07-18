from pathlib import Path
import shutil

import argh
import hydra

from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph
from relational_embeddings.pipeline.graph2text import graph2text
from relational_embeddings.pipeline.text2model import text2model
from relational_embeddings.pipeline.model2emb import model2emb

STAGE2FUNC = {
    "normalize": normalize,
    "table2graph": table2graph,
    "graph2text": graph2text,
    "text2model": text2model,
    "model2emb": model2emb,
}

@argh.arg('overrides', default=None)
def run(wdir, stage=None, *overrides):
    wdir = Path(wdir)
    if stage is None:
        stage = wdir.name
        wdir = wdir.parent
    if overrides is None:
        overrides = []
    if not wdir.is_absolute():
        wdir = Path(__file__).resolve().parent.parent / 'outputs' / wdir
    
    outdir = wdir / stage
    if outdir.exists():
        shutil.rmtree(outdir)

    with hydra.initialize(version_base=None, config_path="../hydra_conf"):
        cfg = hydra.compose(config_name="single_run", overrides=overrides)

    stage_idx = cfg.pipeline.index(stage)
    if stage_idx == 0:
        indir = None
    else:
        indir = wdir / cfg.pipeline[stage_idx - 1]
    run_stage(stage, cfg, outdir, indir=indir)


def run_stage(stage, cfg, outdir, indir=None):
    outdir.mkdir()
    stage_func = STAGE2FUNC[stage]
    stage_func(cfg, outdir, indir=indir)


if __name__ == "__main__":
    argh.dispatch_command(run)
