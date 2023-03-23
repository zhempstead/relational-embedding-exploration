import importlib

from relational_embeddings.lib.utils import get_sweep_vars

def downstream(cfg, outdir, indir=None):
    """
    Evaluate model on downstream ML task (as determined by dataset)
    """
    if indir is None:
        indir = outdir.parent

    teefile = outdir / 'results.txt'

    function = get_pipeline_function("downstream", cfg.downstream.task)
    df = function(outdir, cfg)

    orig_cols = list(df.columns)
    sweep_vars = get_sweep_vars(outdir)
    for var, val in sweep_vars.items():
        df[var] = val
    df = df[list(sweep_vars.keys()) + orig_cols]
    df.to_csv(outdir / 'results.csv', index=False)

    print(f"Done with {cfg.downstream.task}! Results at '{outdir}'")

def get_pipeline_function(pipeline_step, method):
    funcname = f"{method}_{pipeline_step}"
    try:
        module = importlib.import_module(f"relational_embeddings.pipeline.{pipeline_step}.{method}")
        return getattr(module, funcname)
    except ImportError:
        raise ValueError(f"Unrecognized method '{method}' for pipeline step {pipeline_step}")
    except AttributeError:
        raise ValueError(f"There should be a function called {funcname} in '{method}.py' for pipeline step {pipeline_step}")
