from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

from relational_embeddings.lib.token_dict import TokenDict

def main(*files):
    for f in files:
        in_file = Path(f)
        wdir = in_file.parent
        out_file_rows = wdir / (in_file.stem + '_rows.png')
        out_file_norows = wdir / (in_file.stem + '_norows.png')

        tokendict_file = wdir / 'node_dict.feather'
        if not tokendict_file.exists():
            tokendict_file = wdir / 'word_dict.feather'
        if not tokendict_file.exists():
            raise ValueError(f"No token dictionary file at '{wdir}'")
        tokendict = TokenDict()
        tokendict.load(tokendict_file)

        plot_file(in_file, out_file_rows, tokendict, True)
        plot_file(in_file, out_file_norows, tokendict, False)
        print(f"Created plots in {wdir}") 

def plot_file(in_file, out_file, tokendict, keep_row_nodes):
    df = file_freq(in_file, tokendict, keep_row_nodes)
    df.plot(x='rank', y='freq', ylim=[10**-6, 10**-1], title=str(out_file), loglog=True).get_figure().savefig(out_file)

def file_freq(in_file, tokendict, keep_row_nodes):
    counts = defaultdict(int)
    with open(in_file) as fin:
        for line in fin:
            line = line.strip().split()
            for num in line:
                token = tokendict.getTokenForNum(num)
                if token is None:
                    continue
                if not keep_row_nodes and '_row:' in token:
                    continue
                counts[token] += 1
    total_count = sum(counts.values())
    df = pd.DataFrame({
        'token': counts.keys(),
        'freq': [v * 1.0 / total_count for v in counts.values()],
    })
    df = df.sort_values('freq', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    return df

if __name__ == '__main__':
    main(*sys.argv[1:])
