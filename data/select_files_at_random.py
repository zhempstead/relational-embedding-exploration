# This function randomly selected k elements from a specific dataset at random 

import os
import glob
import random 
import sys
import subprocess 

def main():
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '-k':
            k = int(args[i + 1])
        elif arg == '-out_dir':
            out_dir = args[i + 1]
        
    data_file_lst = glob.glob('./chicago/*.csv')
    random_files = random.sample(data_file_lst, k=k)
    # create out directory if it doesn't exist 
    path = "/home/cc/relational-embedding-exploration/data/"
    isExist = os.path.exists(path+"/"+out_dir)
    if not isExist:
        subprocess.run(["mkdir", out_dir])
    # move (or copy) from in directory to out directory 
    for random_file in random_files:
        input_dir = path+random_file
        # print(input_dir)
        subprocess.run(["mv", input_dir, out_dir+"/"])
        
if __name__ == '__main__':
    main()