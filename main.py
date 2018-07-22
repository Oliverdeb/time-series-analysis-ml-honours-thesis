from shapelets.shapelet import Shapelet
from shapelets.shapelet_utils import shapelet_utils
import time
from utils.Utils import Utils

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
    # series = shapelet_utils.percent_diff(series)
    # print (series[:100])
    t = time.time()
    series_cutoff = 600

    _min = 15
    _max = 30
    # print ("before",series)
    # series = shapelet_utils.normalize(series[:series_cutoff])
    # import numpy as np
    # series = [np.log(x) for x in series[:series_cutoff]]
    series = series[:series_cutoff]
    # print ("after", series)
    sets = [series]
    k_shapelets = []
    k = 15
    n_candidates = 15
    for dataset in sets:
        shapelets = []
        for l in range(_min, _max + 1):            
            candidates_i_l = shapelet_utils.generate_candidates(dataset, l)
            prog = len(candidates_i_l)

            print ("Checking candidates of length %d, %d candidates" % (l, prog))
            for i,w in enumerate(candidates_i_l):
                if i % 11 ==0 :
                    print ("\r%.2f" % (i/prog), end="")

                # returns a list of tuples of (distance, corresponding shapelet)
                all_mse = shapelet_utils.find_mse(w, candidates_i_l)

                # sort by distance
                all_mse.sort (key = lambda s: s[0] )
                
                # set quality of this candidate to sum of the best 6 matches / distances
                w.quality = sum (mse for mse, shapelet in all_mse[:n_candidates])

                # store 6 best shapelets that matched
                w.of_same_class = [shapelet for mse, shapelet in all_mse[:n_candidates]]

                # add candidate to list
                shapelets.append( w )
        print()
        # shapelets = shapelet_utils.remove_all_similar(shapelets, (_min + _max) / 4.0)
        shapelets.sort(key = lambda x: x.quality)
        k_shapelets = shapelet_utils.merge(k_shapelets, shapelets)

    def boundary_check(right_shapelet, left_shapelet, thresh):
        if abs(right_shapelet.start_index - left_shapelet.start_index) <= thresh:
            left_shapelet.of_same_class += right_shapelet.of_same_class
            left_shapelet.of_same_class.append(right_shapelet)
            return True
        return False


    print ("Combining shapelets of the same class")
    final = [k_shapelets[0]]
    for i, shapelet in enumerate(k_shapelets[1:]):
        broke = False
        for _shapelet in k_shapelets[:i]:
            # if len(_shapelet.shapelet) != len(shapelet.shapelet):
            #     if boundary_check(_shapelet, shapelet, 5):
            #         broke = True
            #         break
            if shapelet in _shapelet.of_same_class:
                broke = True
                break
        if not broke: final.append(shapelet)
    
    new_final = [final[0]]
    for shapelet in final[1:]:
        flag = False
        for _shapelet in new_final:
            if len(_shapelet.shapelet) != len(shapelet.shapelet):
                if boundary_check(shapelet, _shapelet, 5):
                    flag = True
        if not flag: new_final.append(shapelet)

        
    
    print ("%.2fs elapsed\n%d initial shapelets found\n%d after class check\n%d after combination" % (time.time() -t, len(k_shapelets), len(final), len(new_final)))

    with open('out.csv', 'w') as f:
        for i,shapelet in enumerate(new_final):
            f.write(str(i) + "," + shapelet.tocsv() + "\n")
            for similar in shapelet.of_same_class:
                f.write(str(i) + "," + similar.tocsv() + "\n")

    # shapelet_utils.graph(series[:series_cutoff], final[:k])
    shapelet_utils.graph_classes(new_final[:k])



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