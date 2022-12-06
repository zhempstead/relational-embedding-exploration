from pathlib import Path
import subprocess
import sys

import pandas as pd

from relational_embeddings.lib.utils import get_sweep_vars, prev_stage_dir

def main(experiment_dir):
    dfs = []
    for textfile in experiment_dir.rglob('text.txt'):
        stage_dir = textfile.parent
        sweep_vars = get_sweep_vars(stage_dir)
        df = pd.DataFrame({k: [v] for k, v in sweep_vars.items()})

        cmd = ['wc', '-w', textfile]
        wc = int(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8').strip().split()[0])
        df['corpus_size'] = wc
        
        dfs.append(df)

    full_df = pd.concat(dfs)
    results = experiment_dir / 'results.csv'
    res_df = pd.read_csv(results)
    res_df = res_df.merge(full_df)
    res_df.to_csv(results, index=False)

    print(f"Added corpus_size to '{results}'")

if __name__ == '__main__':
    main(Path(sys.argv[1]))
