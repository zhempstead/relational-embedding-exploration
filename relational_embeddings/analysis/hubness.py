from pathlib import Path
import sys

from gensim.models import KeyedVectors
import pandas as pd
from skhubness import Hubness
from tqdm import tqdm

from relational_embeddings.lib.utils import get_sweep_vars

def main(experiment_dir):
    outfile_rows = experiment_dir / 'hubness_rows.csv'
    outfile_all = experiment_dir / 'hubness_rows.csv'

    rows_records = []
    print("Row embeddings:")
    for infile in tqdm(list(experiment_dir.rglob('embeddings.csv'))):
        wdir = infile.parent
        X = pd.read_csv(infile)
        rows_records.append(get_hubness_row(wdir, X))

    df = pd.DataFrame.from_records(rows_records)
    df['scope'] = 'base_rows'
    df.to_csv(outfile_rows, index=False)
    print(f"Wrote csv to {outfile_rows}") 

    all_records = []
    print("Total embeddings:")
    for infile in tqdm(list(experiment_dir.rglob('model'))):
        wdir = infile.parent
        model = KeyedVectors.load_word2vec_format(infile)
        X = model.vectors
        all_records.append(get_hubness_row(wdir, X))

    df = pd.DataFrame.from_records(all_records)
    df['scope'] = 'all'
    df.to_csv(outfile_all, index=False)
    print(f"Wrote csv to {outfile_all}") 


def get_hubness_row(wdir, X):
    row = get_sweep_vars(wdir)
    hub = Hubness(k=10, return_value="all", metric='cosine')
    hs = hub.fit(X).score()
    for metric in hs.keys():
        row[metric] = hs[metric]
    return row

if __name__ == '__main__':
    main(Path(sys.argv[1]))
