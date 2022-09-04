from pathlib import Path
import shutil

import argh
import hydra

from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph
from relational_embeddings.pipeline.graph2model import graph2model
from relational_embeddings.pipeline.graph2text import graph2text
from relational_embeddings.pipeline.text2model import text2model
from relational_embeddings.pipeline.model2emb import model2emb
from relational_embeddings.pipeline.classification import classification

STAGE2FUNC = {
    "normalize": normalize,
    "table2graph": table2graph,
    "graph2model": graph2model,
    "graph2text": graph2text,
    "text2model": text2model,
    "model2emb": model2emb,
    "classification": classification,
}

@argh.arg('overrides', default=None)
def run(wdir, stage=None, *overrides):
    wdir = Path(wdir)
    is_multirun = 'multirun' in wdir.parts
    if stage is None:
        stage = wdir.name.split(',')[0]
        if not is_multirun:
            wdir = wdir.parent
    if overrides is None:
        overrides = []
    if is_multirun:
        overrides += ('multirun=True',)

    if is_multirun:
        outdir = wdir
    else:
        outdir = wdir / stage
        if outdir.exists():
            shutil.rmtree(outdir)

    with hydra.initialize(version_base=None, config_path="../hydra_conf"):
        cfg = hydra.compose(config_name="single_run", overrides=overrides)
    stage_idx = cfg.pipeline.index(stage)
    if stage_idx == 0:
        indir = None
    elif is_multirun:
        indir = wdir.parent
    else:
        indir = wdir / cfg.pipeline[stage_idx - 1]
    run_stage(stage, cfg, outdir, indir=indir)


def run_stage(stage, cfg, outdir, indir=None):
    outdir.mkdir(exist_ok=True)
    stage_func = STAGE2FUNC[stage]
    stage_func(cfg, outdir, indir=indir)


if __name__ == "__main__":
    argh.dispatch_command(run)
