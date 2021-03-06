import numpy as np
import os
from pandas import read_csv

files = {}

archive_dir = '../data/jse'

def analyze_sets():
    """
    Perform basic analysis on datasets in archive dir above, extract most, middle and least
    volatile stocks by variance.

    Plots the stocks.
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
    for i,key in enumerate(list_files[:3]):
        legend = key.split('-')[1]    
        plt.plot(sets[key].values, label='%d MOST variable %s: %.2f' % (i+1, legend, files[key]['variance']))    
        print ('%s:' % key, files[key])

    k = list_files[len(list_files) // 2]
    print ("\nMiddle ish:\n%s"%k, files[k])
    print ("\n\nTop 3 LEAST variable stocks (most stable?)")    

    for i,key in enumerate(list_files[:-4:-1]):
        legend = key.split('-')[1]
        plt.plot(sets[key].values, label='%d LEAST var %s: %.2f' % (3-i, legend, files[key]['variance']))
        print ('%s:' % key, files[key])
        
    hehe =len(list_files)//2
    for key in list_files[hehe-2:hehe+1]:
        plt.plot(sets[key].values, label='%d normal var %s: %.2f' % (3-i, legend, files[key]['variance']))

    # most = list_files[0]
    # least = list_files[-1]
    plt.legend(loc='best')
    plt.show()
    




def remove_below_n_lines(n):
    """
    Remove datasets with less than N datapoints
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