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


    _min = 15
    _max = 16
    sets = [series[:100]]
    k_shapelets = []
    k = 10
    for dataset in sets:
        shapelets = []

        for l in range(_min, _max):
            W_i_j = util.generate_candidates(dataset, _min, _max)
            # print ("len candidates", len(W_i_j))
            # print ("empty is ", [] in W_i_j)

            for w in W_i_j:
                # print ("candidate", w)
                all_mse = util.find_mse(w.shapelet, W_i_j)
                all_mse.sort()
                quality = sum ( all_mse[:6])
                shapelets.append( (w, quality) )

        shapelets.sort(key = lambda x: x[1])
        # TODO: remove similar, ie overlap and from same series
        k_shapelets = util.merge(k, k_shapelets, shapelets)
    
    print (k_shapelets)
    util.graph(series[:100], k_shapelets)



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