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
def run(outdir, *overrides):
    outdir = Path(outdir)
    stage = outdir.name.split(',')[0]
    if overrides is None:
        overrides = []

    with hydra.initialize(version_base=None, config_path="../hydra_conf"):
        cfg = hydra.compose(config_name="run", overrides=overrides, return_hydra_config=True)
    stage_idx = cfg.hydra.sweeper.pipeline.index(stage)
    if stage_idx == 0:
        indir = None
    else:
        indir = outdir.parent
    run_stage(stage, cfg, outdir, indir=indir)


def run_stage(stage, cfg, outdir, indir=None):
    outdir.mkdir(exist_ok=True)
    stage_func = STAGE2FUNC[stage]
    stage_func(cfg, outdir, indir=indir)


if __name__ == "__main__":
    argh.dispatch_command(run)
