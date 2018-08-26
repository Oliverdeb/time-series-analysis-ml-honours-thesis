import numpy as np
import os
from pandas import read_csv

files = {}

archive_dir = '../data/jse'


 for _file in os.listdir(archive_dir):
        print (_file)
        open_file = readcsv(os.path.join(archive_dir, _file), 



def remove_below_n_lines(n):
    for _file in os.listdir(archive_dir):
        print (_file)
        _file_ = open(os.path.join(archive_dir, _file), 'rb')
        lines = len(_file_.readlines())
        _file_.close()
        if lines < n:
            os.remove(os.path.abspath(os.path.join(archive_dir, _file)))

    #     if not _file.endswith('.csv'):
    #         continue
        
    #     files[os.path.join(archive_dir, _file)] = {
    #         'variance' : np.var()
    #     }
