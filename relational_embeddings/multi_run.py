from pathlib import Path

import hydra

from relational_embeddings.single_run import STAGE2FUNC

from relational_embeddings.pipeline.normalize import normalize
from relational_embeddings.pipeline.table2graph import table2graph


@hydra.main(version_base=None, config_path="../hydra_conf", config_name="multi_run")
def run(cfg):
    """
    Build embedding and evaluate on downstream model
    """
    stage_func = STAGE2FUNC[cfg.pipeline_stage]
    stage_func(cfg, Path.cwd())


if __name__ == "__main__":
    run()
