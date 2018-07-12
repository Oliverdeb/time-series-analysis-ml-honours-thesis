from utils.Utils import Utils
from shapelets.shapelet import Shapelet
from shapelets.util import util

def main():
    # shapelet_classifier = Shapelet()

    series = [float(x) for x in open('data/snp2.csv').readlines()]
    # print (series[:100])
    # exit(1)
    # a = [ (1,2), (2,5), (1,1), (1,9)]
    # b = [ (1,3)] #, (1,1), (1,15), (1,8)]
    # b.sort(key = lambda x: x[1])
    # print ('a' , [arr[1] for arr in a])
    # print ('b' , [arr[1] for arr in b])
    # k = Shapelet.merge(4, a, b)
    # print ('a' , [arr[1] for arr in a])
    # print ('b' , [arr[1] for arr in b])

    # print (k)

    series_cutoff = 300

    _min = 15
    _max = 16
    # print ("before",series)
    # series = util.normalize(series[:series_cutoff])
    # import numpy as np
    # series = [np.log(x) for x in series[:series_cutoff]]
    series = series[:series_cutoff]
    # print ("after", series)
    sets = [series]
    k_shapelets = []
    k = 20
    for dataset in sets:
        shapelets = []

        for l in range(_min, _max + 1):
            W_i_j = util.generate_candidates(dataset, _min, _max)

            for w in W_i_j:
                all_mse = util.find_mse(w, W_i_j)
                all_mse.sort()
                w.quality = sum ( all_mse[:6])
                shapelets.append( w )

        shapelets = util.remove_all_similar(shapelets, (_min + _max) / 4.0)
        shapelets.sort(key = lambda x: x.quality)
        k_shapelets = util.merge(k, k_shapelets, shapelets)
    
    print ("%d shapelets found" % (len(k_shapelets)))
    util.graph(series[:series_cutoff], k_shapelets[:k])



if __name__ == '__main__':  
    main()
    exit(1)
    print('\n===== POINTS =====')
    points = Utils.read_file('data/snp2.csv')
    print(len(points))

    points = Utils.moving_average(points)
    print(len(points))
    Utils.print_array(points)

    print('\n===== LINES =====')
    lines = Utils.sliding_window_segmentation(points, 120)
    Utils.print_array(lines)
    print('\n')

    p = []
    for x in points:
        p.append(x.price)

    Utils.graph(p, lines)