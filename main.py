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
    series_cutoff = 1000
    import numpy as np
    before = series[0]

    _mean = np.mean(series)
    _sd = np.std(series)
    _min = 20
    _max = 30
    # print ("before",series)
    # series = shapelet_utils.normalize(series[:series_cutoff])
    # series = [np.log(x) for x in series[:series_cutoff]]
    # print ("after", series)
    series = series[:series_cutoff]

    sets = [series]
    k_shapelets = []
    k = 10
    n_candidates = 15
    mse_threshold = 20
    print ("mse %.4f, mean %.4f, sd %.4f" % (mse_threshold, _mean, _sd))
    print ("before is %.2f, after is %.2f" % (before, series[0]))
    # mse_threshold = mse_threshold / (before / series[0])
    # mse_threshold = 0.5
    print ("new mse %.4f" % mse_threshold)
    
    quality_threshold = True
    for dataset in sets:
        shapelets = []
        # for l in range(_min, _max + 1):            
        # candidates_i_l = shapelet_utils.generate_candidates(dataset, l)
        candidates_i_l = shapelet_utils.generate_all_size_candidates(dataset, _min, _max + 1)
        prog = len(candidates_i_l)

        print ("\r\tChecking candidates of length %d, %d candidates:" % (_min, prog), end="")
        for i,w in enumerate(candidates_i_l):
            if i % 11 ==0 :
                print ("\r%.2f" % (i/prog), end="")

            w.of_same_class = shapelet_utils.find_new_mse(w, candidates_i_l[:i] + candidates_i_l[i+1:], mse_threshold)
            # print (len(w.of_same_class))
            w.quality = len(w.of_same_class)
            shapelets.append( w )

            continue
            # returns a list of tuples of (MSE, corresponding shapelet)
            mse_and_shapelets = shapelet_utils.find_mse(w, candidates_i_l[:i] + candidates_i_l[i+1:])
            
            # print ("mse", mse_and_shapelets[0][0])
            for mse,shapelet in mse_and_shapelets:
                if mse == 0: print("MSE OF 0")
            if quality_threshold:
                # add all shapelets to a set that are below the threshold MSE value
                w.of_same_class = {shapelet for mse, shapelet in mse_and_shapelets if mse <= mse_threshold}

                # set quality to length of the set of shapelets of the same class (ie. number of shapelets <= threshold)
                w.quality = len(w.of_same_class)
            else:
                # sort by MSE        
                mse_and_shapelets.sort(key = lambda x: x[0])

                # set quality of this candidate to sum of the best N matches / MSE
                w.quality = sum (mse for mse, shapelet in mse_and_shapelets[:n_candidates])

                # store N best shapelets that matched in a set
                w.of_same_class = {shapelet for mse, shapelet in mse_and_shapelets[:n_candidates]}

            # add candidate to list
            shapelets.append( w )
        print()
        # shapelets = shapelet_utils.remove_all_similar(shapelets, (_min + _max) / 5.0)
        shapelets.sort(key = lambda x: x.quality, reverse=quality_threshold)
        k_shapelets = shapelet_utils.merge(k_shapelets, shapelets)

    print ("time taken to get all %.2f" % (time.time() - t))    
   
    final = shapelet_utils.remove_duplicates(k_shapelets)
    
    print ("%.2fs elapsed\n%d initial shapelets found\n%d after class check" % (time.time() -t, len(k_shapelets), len(final)))

    
    file_name = '-'.join((str(series_cutoff), str(_min), str(_max), str(mse_threshold), str(len(final)))) + '.csv'
    with open(file_name, 'w') as f:
        f.write("target,sequence\n")
        for i,shapelet in enumerate(final):
            f.write(str(i) + "," + shapelet.to_csv_offset_0() + "\n")
            for similar in shapelet.of_same_class:
                f.write(str(i) + "," + similar.to_csv_offset_0() + "\n")

    shapelet_utils.graph_classes(final[:k], series[:series_cutoff])

    # shapelet_utils.graph(series[:series_cutoff], final[:k])
    
    # final.sort(key = lambda x: x.quality, reverse=quality_threshold)

    
    



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