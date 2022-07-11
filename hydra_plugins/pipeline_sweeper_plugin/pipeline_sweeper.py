# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hydra.types import HydraContext
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig, OmegaConf

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside sweep()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.

log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    pipelines: List[List[str]]
    stage_variants: Dict[str, List[str]]
    _target_: str = (
        "hydra_plugins.pipeline_sweeper_plugin.pipeline_sweeper.PipelineSweeper"
    )


ConfigStore.instance().store(group="hydra/sweeper", name="pipeline", node=LauncherConfig)


class PipelineSweeper(Sweeper):
    def __init__(self, pipelines: List[List[str]], stage_variants: Dict[str, List[str]]):
        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None
        self.pipelines = pipelines
        self.stage_variants = stage_variants

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )
        self.hydra_context = hydra_context

    def sweep(self, arguments: List[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        log.info(f"Sweep output dir : {self.config.hydra.sweep.dir}")

        # Save sweep run config in top level sweep working directory
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        pipeline_subsets = set()
        tasks = []
        for pipeline in self.pipelines:
          pipeline_subset = []
          lists = []
          for stage in pipeline:
            pipeline_subset.append(stage)
            sweep = [f"{stage}={variant}" for variant in self.stage_variants[stage]]
            lists.append(sweep)
            if tuple(pipeline_subset) not in pipeline_subsets:
              pipeline_subsets.add(tuple(pipeline_subset))
              tasks += itertools.product([f"+pipeline_stage={stage}"], *lists)

        tasks = [add_pipeline_subdir(task) for task in tasks]

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        simple_overrides = []
        for override in parsed:
            if override.is_sweep_override():
                # Sweepers must manipulate only overrides that return true to is_sweep_override()
                # This syntax is shared across all sweepers, so it may limiting.
                # Sweeper must respect this though: failing to do so will cause all sorts of hard to debug issues.
                # If you would like to propose an extension to the grammar (enabling new types of sweep overrides)
                # Please file an issue and describe the use case and the proposed syntax.
                # Be aware that syntax extensions are potentially breaking compatibility for existing users and the
                # use case will be scrutinized heavily before the syntax is changed.
              raise ValueError("Sweep overrides incompatible with pipeline sweeper")
                #sweep_choices = override.sweep_string_iterator()
                #key = override.get_key_element()
                #sweep = [f"{key}={val}" for val in sweep_choices]
                #lists.append(sweep)
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                simple_overrides.append(f"{key}={value}")

        tasks = [add_overrides(task, simple_overrides) for task in tasks]
        self.validate_batch_is_legal(tasks)
        return self.launcher.launch(tasks, initial_job_idx=0)

def add_pipeline_subdir(task: Tuple[str, ...]) -> Tuple[str, ...]:
  subdir = '/'.join(task[1:])
  return add_overrides(task, [f"+pipeline_subdir='{subdir}'"])

def add_overrides(task: Tuple[str, ...], overrides: List[str]) -> Tuple[str, ...]:
  return tuple(list(task) + overrides)
