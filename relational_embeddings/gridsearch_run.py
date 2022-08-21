import os
from pathlib import Path

import hydra

from relational_embeddings.stage_run import STAGE2FUNC

from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph


@hydra.main(version_base=None, config_path="../hydra_conf", config_name="gridsearch_run")
def run(cfg):
    """
    Build embedding and evaluate on downstream model
    """
    print(cfg)
    if cfg.resume_workdir is not None:
        resume_subdir = Path(cfg.resume_workdir) / cfg.pipeline_subdir
        resume_subdir.mkdir(exist_ok=True)
        os.chdir(resume_subdir)
        print(f"Subdir changed to '{Path.cwd()}'")

    donefile = Path.cwd() / 'DONE'
    if donefile.exists():
        print(f"{donefile} already exists... skipping this job")
        return
    stage_func = STAGE2FUNC[cfg.pipeline_stage]
    stage_func(cfg, Path.cwd())
    donefile.touch()


if __name__ == "__main__":
    run()
