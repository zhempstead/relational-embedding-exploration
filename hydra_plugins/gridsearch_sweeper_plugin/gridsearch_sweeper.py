# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import itertools
import logging
from pathlib import Path
import shutil
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
    pipeline: List[str]
    _target_: str = (
        "hydra_plugins.gridsearch_sweeper_plugin.gridsearch_sweeper.GridsearchSweeper"
    )


ConfigStore.instance().store(group="hydra/sweeper", name="gridsearch", node=LauncherConfig)


class GridsearchSweeper(Sweeper):
    def __init__(self, pipeline: List[str]):
        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None
        self.pipeline = pipeline

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


        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        simple_overrides = []

        stage_sweeps = {stage: [] for stage in self.pipeline}

        for override in parsed:
            if override.is_sweep_override():
                # Sweepers must manipulate only overrides that return true to is_sweep_override()
                # This syntax is shared across all sweepers, so it may limiting.
                # Sweeper must respect this though: failing to do so will cause all sorts of hard to debug issues.
                # If you would like to propose an extension to the grammar (enabling new types of sweep overrides)
                # Please file an issue and describe the use case and the proposed syntax.
                # Be aware that syntax extensions are potentially breaking compatibility for existing users and the
                # use case will be scrutinized heavily before the syntax is changed.
                sweep_choices = override.sweep_string_iterator()
                key = override.get_key_element()
                stage = key.split('.')[0]
                if stage == 'global':
                    stage = self.pipeline[0]
                if stage not in stage_sweeps:
                    raise ValueError(f"Sweep over '{key}' doesn't correspond to a pipeline stage")
                stage_sweeps[stage].append((key, sweep_choices))
            else:
                key = override.get_key_element()
                value = override.get_value_element_as_str()
                if key == "resume_workdir":
                    shutil.copy(sweep_dir / 'multirun.yaml', value)
                simple_overrides.append(f"{key}={value}")


        tasks = {}
        prev_stage = None
        returns = []
        initial_job_idx = 0
        for stage in self.pipeline:
            stage_overrides = []
            for key, values in stage_sweeps[stage]:
                stage_overrides.append([(key, value) for value in values])
            if stage_overrides:
                stage_tasks = [(stage, list(overrides)) for overrides in itertools.product(*stage_overrides)]
            else:
                stage_tasks = [(stage, [])]
            if prev_stage is None:
                tasks[stage] = [[st] for st in stage_tasks]
            else:
                tasks[stage] = [list(prev) + [new] for prev, new in itertools.product(tasks[prev_stage], stage_tasks)]
            prev_stage = stage
            batch = [task2hydra(t, simple_overrides) for t in tasks[stage]]
            self.validate_batch_is_legal(batch)
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)
            initial_job_idx += len(batch)
            returns.append(results)

        return returns

def task2hydra(task: List[Tuple[str, List[Tuple[str, str]]]], simple_overrides: List[str]):
    subdir_path = []
    all_overrides = simple_overrides.copy()
    for stage, overrides in task:
        subdir_path.append(get_subdir(stage, overrides))
        if overrides is not None:
            all_overrides += [f"{key}={value}" for key, value in overrides]
    all_overrides += [f"+pipeline_stage={stage}", f"+pipeline_subdir='{'/'.join(subdir_path)}'"]
    return tuple(all_overrides)

def get_subdir(stage: str, overrides: List[Tuple[str, str]]):
    '''
    Return the subdir for the given stage and set of overrides
    '''
    if not len(overrides):
        return stage
    subdir = [stage]
    for key, value in overrides:
        if key == stage:
            subdir.append(value)
            continue
        key_prefix, key_suffix = key.split('.', 1)
        if key_prefix == 'global':
            outdir_key = key
        else:
            outdir_key = key_suffix
        try:
            # So directories of int overrides have nice sort order
            subdir.append(f"{outdir_key}={int(value):04d}")
        except ValueError:
            subdir.append(f"{outdir_key}={value}")
    return ",".join(subdir)


def add_overrides(task: Tuple[str, ...], overrides: List[str]) -> Tuple[str, ...]:
  return tuple(list(task) + overrides)
