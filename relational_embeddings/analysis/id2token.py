from pathlib import Path
import sys

from relational_embeddings.lib.token_dict import TokenDict

def main(*files):
    for f in files:
        in_file = f
        out_file = in_file + '.tokens'

        wdir = Path(in_file).parent
        tokendict_file = wdir / 'node_dict.feather'
        if not tokendict_file.exists():
            tokendict_file = wdir / 'word_dict.feather'
        if not tokendict_file.exists():
            raise ValueError(f"No token dictionary file at '{wdir}'")
        tokendict = TokenDict()
        tokendict.load(tokendict_file)
            
        file_id2token(in_file, out_file, tokendict)
        print(f"Tokenized '{in_file}' at '{out_file}'")

def file_id2token(in_file, out_file, tokendict):
    with open(in_file) as fin, open(out_file, 'w') as fout:
        for line in fin:
            fout.write(line_id2token(line, tokendict))

def line_id2token(line_in, tokendict):
    line = line_in.strip().split()
    line = [tokendict.getTokenForNum(item) or item for item in line]
    return ' '.join(line) + '\n'

if __name__ == '__main__':
    main(*sys.argv[1:])
