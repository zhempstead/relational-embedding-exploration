import os
import glob

def main():
    data_file_lst = glob.glob('./chicago_4/*.csv')
    for data_file in data_file_lst:
        data = []
        with open(data_file) as f:
            for line in f:
                item = line.strip()
                if item == '':
                    continue

                data.append(item)
        
        if len(data) == 0:
            os.remove(data_file) 
            
            print('delete %s' % data_file)


if __name__ == '__main__':
    main()