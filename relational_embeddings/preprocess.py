import shutil

import hydra
import pandas as pd

from relational_embeddings.lib.utils import all_csv_in_path, dataset_dir, make_symlink


@hydra.main(version_base=None, config_path="../hydra_conf", config_name="preprocess")
def preprocess(cfg):
    """
    - Make copies of target file with target column removed
    - Remove target column from all other files
    """
    target_column = cfg.dataset.target_column
    location = dataset_dir(cfg.dataset.name)
    base_file = location / "base.csv"

    train_emb_dir = location / "train_embeddings"
    if train_emb_dir.exists():
        shutil.rmtree(train_emb_dir)
    train_emb_dir.mkdir()

    df = pd.read_csv(base_file)
    df = df[df[target_column].notna()]

    x_file = location / "base_x.csv"
    y_file = location / "base_y.csv"

    y_df = df[target_column]
    x_df = df.drop(target_column, axis=1)

    for df, fname in [
        (x_df, x_file),
        (y_df, y_file),
    ]:
        print(f"Writing '{fname}'...")
        df.to_csv(fname, index=False)

    make_symlink(x_file, train_emb_dir / "base.csv")
        

    for path in all_csv_in_path(location, exclude_base=True):
        df = pd.read_csv(path, sep=",", low_memory=False)
        if target_column[0] in df.columns:
            print(f"Dropping column '{target_column}' from '{path}'")
            df = df.drop(columns=target_column)
            df.to_csv(path, index=False)

        make_symlink(path, train_emb_dir / path.name)


if __name__ == "__main__":
    preprocess()
