from pathlib import Path

import hydra
import pandas as pd

from relational_embeddings.lib.utils import all_data_files_in_path

@hydra.main(version_base=None, config_path='../hydra_conf', config_name='preprocess')
def preprocess(cfg):
    '''
    - Make copy of target file with target column removed
    - Remove target column from all other files
    '''
    name = cfg.dataset.name
    target_column = cfg.dataset.target_column
    location = Path(__file__).resolve().parent.parent / 'data' / name
    base_file = location / 'base.csv'
    processed_file = location / 'base_processed.csv'

    df = pd.read_csv(base_file)
    if target_column in df.columns:
        print(f"Dropping rows from '{base_file}' where '{target_column}' is NA")
        df = df[df[target_column].notna()]
        df.to_csv(base_file, index=False)
        df = df.drop(columns=[target_column])
        print(f"Writing '{processed_file}' without '{target_column}'")
    else:
        print("Writing '{processed_file}'")
    df.to_csv(processed_file, index=False)

    for path in all_data_files_in_path(location):
        df = pd.read_csv(path, encoding = 'latin1', sep=',', low_memory=False)
        if target_column[0] in df.columns: 
            print(f"Dropping column '{target_column}' from '{path}'")
            df = df.drop(columns = target_column)
            df.to_csv(path, index=False)

if __name__ == "__main__":
    preprocess()
