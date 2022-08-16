from pathlib import Path
import sys

import pandas as pd
from skhubness import Hubness

from relational_embeddings.lib.utils import get_sweep_vars

def main(experiment_dir):
    outfile = experiment_dir / 'hubness.csv'

    rows = []
    for infile in experiment_dir.rglob('embeddings.csv'):
        wdir = infile.parent
        X = pd.read_csv(infile)
        rows.append(get_hubness_row(wdir, X))

    df = pd.DataFrame.from_records(rows)
    df.to_csv(outfile, index=False)
    print(f"Wrote csv to {outfile}") 

def get_hubness_row(wdir, X):
    row = get_sweep_vars(wdir)
    hub = Hubness(k=10, return_value="all", metric='cosine')
    hs = hub.fit(X).score()
    for metric in ['k_skewness', 'hub_occurrence', 'antihub_occurrence']:
        row[metric] = hs.get(metric)
    return row

if __name__ == '__main__':
    main(Path(sys.argv[1]))
