import os
import glob
import sys

import pandas as pd 

def main():
    args = sys.argv[1:]
    target_dir = args[0]
    data_file_lst = glob.glob('./'+target_dir+'/*.csv')
    total_rows = 0
    for data_file in data_file_lst:
        results = pd.read_csv(data_file)
        total_rows += len(results)
    print(f"There are {total_rows} rows")
        

if __name__ == '__main__':
    main()