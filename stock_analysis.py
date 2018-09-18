import numpy as np
import os
from pandas import read_csv

files = {}

archive_dir = 'data/jse'

def analyze_sets():
    """
    Function that performs basic stats on the datasets and outputs the least, middle and most variable stocks.
    """
    files = {}
    sets = {}
    for _file in os.listdir(archive_dir):
        path = os.path.join(archive_dir, _file)
        df = read_csv(path, usecols=[1]) 
        sets[_file] = df
        files[_file] = {
             'variance' : df['Closing (c)'].std(),
             'mean' : df['Closing (c)'].mean(),
        }

    # list_files = [(k,v) for k,v in sorted(files.items(), key=lambda (i,j): (j, i))]
    list_files = sorted(files, key=lambda x: (files[x]['variance'], files[x]['mean']) , reverse=True)
    
    import matplotlib.pyplot as plt

    print ("Top 3 most variable stocks")
    for i,key in enumerate(list_files[:1]):
        legend = key.split('-')[1]    
        plt.plot(list(reversed(sets[key].values))[-2000:], label=legend)    
        # plt.plot(sets[key].values[:2500], label=legend)    
        print ('%s:' % key, files[key])

    print ("\n\nMiddle")    
    leng = int(len(list_files)/2)
    for i,key in enumerate(list_files[leng:leng+1]):
        legend = key.split('-')[1]
        plt.plot(list(reversed(sets[key].values))[-2000:], label=legend)
        # plt.plot(sets[key].values[:2500], label=legend)    

        print ('%s:' % key, files[key])
    
    print ("\n\nTop 3 least variable stocks (most stable?)")    
    for i,key in enumerate(list_files[-1:]):
        legend = key.split('-')[1]
        plt.plot(list(reversed(sets[key].values))[-2000:], label=legend)
        # plt.plot(sets[key].values[:2500], label=legend)    

        print ('%s:' % key, files[key])
    # hehe =len(list_files)//2
    # for key in list_files[hehe-2:hehe+1]:
    #     plt.plot(sets[key].values, label='%d normal var %s: %.2f' % (3-i, legend, files[key]['variance']))

    # most = list_files[0]
    # least = list_files[-1]
    plt.xlabel("Time (days)")
    plt.ylabel("Price (ZAR)")
    plt.legend(loc='best')
    plt.show()
    




def remove_below_n_lines(n):
    """
    Function that removes datasets with less than N data points.
    """
    for _file in os.listdir(archive_dir):
        print (_file)
        _file_ = open(os.path.join(archive_dir, _file), 'rb')
        lines = len(_file_.readlines())
        _file_.close()
        if lines < n:
            os.remove(os.path.abspath(os.path.join(archive_dir, _file)))



if __name__ == '__main__':
    analyze_sets()