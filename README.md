
# Relational Embedding for Downstream Machine Learning Tasks

  
## Introduction

We apply relational embedding to boost the performance of machine learning models. The relational embedding captures information that exists across multiple tables and could attain a similar level of performance in downstream models with a minimal amount of human effort.

 
## Setup 
To setup the environment, run, 
```
git submodule update --init rwalk/
git submodule update --init scikit-hubness/

virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt
pip install -e scikit-hubness
pip install -e .
(cd rwalk/rwalk; make all)
```

## Datasets

A dataset should be placed in its own folder under `dataset`. A corresponding config file needs to be created under `hydra_conf/dataset`. Datasets should consist of csv files, and the file containing the column used as Y for the downstream task should be called 'base.csv'.

## Configuration

This project uses [Hydra](https://github.com/facebookresearch/hydra) for configuration. All configuration lives under `hydra_conf`.

All configuration can be overridden on the command line, or multiple values can be specified to sweep over all combinations in parallel. Examples will follow in a later section.

## Architecture

`relational_embeddings/gridsearch_run.py` lets us run an arbitrary pipeline (arbitrary in the sense that we can add whatever steps we want), with arbitrary variants for each pipeline step.

`hydra_conf/run.yaml` specifies the defaults for both the pipeline (i.e. which steps get run) and also each pipeline step (i.e. when running the `table2graph` pipeline step we can either create a leva-style graph or an embDI-style graph). The default pipeline is currently `leva_rwalk`, which is specified at `hydra_conf/pipeline/leva_rwalk`.

The code for pipeline steps lives under `relational_embeddings/pipeline`. See `relational_embeddings/pipeline/table2graph` for a working example of a pipeline step with multiple variants.

## Example

Run the following command:
```
source venv/bin/activate
PYTHONHASHSEED=0 python relational_embeddings/gridsearch_run.py dataset=genes graph2text.walk_length=20 text2model.iter=4 text2model.dimensions=8,16,64 model2emb.use_value_nodes=True,False
```
- `PYTHONHASHSEED=0` is necessary to make the pipeline fully deterministic modulo the random seed (which can be overridden via setting `normalize.random_seed`)
- We are overriding the dataset and several step-specific parameters
- In two cases (`text2model.dimensions` and `model2emb.use_value_nodes`) we are setting multiple values. This will make the script run that pipeline step (and all downstream steps) for all relevant combinations of the values. So there will be 3 different runs of `text2model` and 6 different runs of `model2emb`. Multiple runs within a pipeline step run in parallel.

Once this finishes running, the output will be placed under `multirun/` (further nested by date and timestamp). The output directory for each pipeline step is nestled under its parent.

Afterwards, it might make sense to run some analysis scripts on the data.

```
# Gather the final downstream output into one csv file
python relational_embeddings/analysis/gather_results.py multirun/<date>/<timestamp>

# Calculate hubness metrics for all the different embeddings
python relational_embeddings/analysis/hubness.py multirun/<date>/<timestamp>
```

## Debugging / Resuming

Unfortunately, running the full pipeline will elide errors. If an early step fails then later steps are likely to fail instantly with no error message (since they are expecting input that doesn't exist). You can see which jobs finished successfully by checking the output directory for the presence of an empty file called `DONE`.

If a job isn't working as expected, you can debug it by running
```
python relational_embeddings/stage_run.py multirun/<date>/<timestamp>/<path to output directory of step to debug>
```
Once you have figured out what went wrong, you can resume the original full pipeline run by rerunning the original command with `resume_workdir=multirun/<date>/<timestamp>`. It will still create a new output directory (this is a limitation of the Hydra framework), but will actually resume work on the original output directory. When you rerun, it will skip any jobs marked with the `DONE` file. You can take advantage of this behavior in other ways - for instance, suppose your jobs completed successfully but the output was invalid due to a bug. Simply remove the `DONE` files in the jobs you want to rerun (including any downstream jobs) and rerun the command with `resume_workdir` specified.

## Overview of pipeline steps

### Leva-style random walks

- normalize: tokenize, handle whitespace, etc. convert numerical columns into discrete bins (treated as strings).
- table2graph: convert the table to a graph with nodes for each value, row, and (sometimes) column. Also create a dictionary mapping tokens to integers (since it's easier to deal with a graph with integer labels)
- graph2text: do random walks on the graph to produce "text".
- text2model: train on the text to produce an embeddings model.
- model2emb: Produce concrete embeddings for each row in base.csv. The embeddings are either the embedding of the row token alone, or the sum of the embeddings of the row token and each value token in the node.
- downstream: Train a simple model using the embeddings as X and the target_column originally from base.csv as Y. Report the training and test accuracy. In general we split the data into 5 groups. Each group is used as the test set once (the model is re-trained 5 times), and then we report the average trainging/test accuracy over all 5 splits.

### Leva-style matrix factorization

- normalize: same as above
- table2graph: same as above
- graph2model: Create the model directly from the graph by converting the edgelist into an adjacency matrix and factorizing.
- model2emb: same as above
- downstream: same as above
