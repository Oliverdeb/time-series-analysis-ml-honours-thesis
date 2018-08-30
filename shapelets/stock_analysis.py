import numpy as np
import os
from pandas import read_csv

files = {}

archive_dir = '../data/jse'

def analyze_sets():
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
    
    print ("\n\nTop 3 LEAST variable stocks (most stable?)")    
    for i,key in enumerate(list_files[-3:]):
        legend = key.split('-')[1]
        plt.plot(sets[key].values, label='%d LEAST var %s: %.2f' % (3-i, legend, files[key]['variance']))
        print ('%s:' % key, files[key])
    
    # most = list_files[0]
    # least = list_files[-1]
    plt.legend(loc='best')
    plt.show()
    




def remove_below_n_lines(n):
    for _file in os.listdir(archive_dir):
        print (_file)
        _file_ = open(os.path.join(archive_dir, _file), 'rb')
        lines = len(_file_.readlines())
        _file_.close()
        if lines < n:
            os.remove(os.path.abspath(os.path.join(archive_dir, _file)))



if __name__ == '__main__':
    analyze_sets()