from pathlib import Path
import sys

import pandas as pd

def main(experiment_dir):
    outfile = experiment_dir / 'results.csv'
    dfs = [pd.read_csv(r) for r in experiment_dir.rglob('results.csv')]
    df = pd.concat(dfs)
    df.to_csv(outfile, index=False)
    print(f"Wrote results from {len(dfs)} files to '{outfile}'")

if __name__ == '__main__':
    main(Path(sys.argv[1]))
