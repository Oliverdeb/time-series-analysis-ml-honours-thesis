from shapelets.shapelet import Shapelet
from shapelets.shapelet_utils import shapelet_utils
# from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import time, argparse

def main():

    series = [float(x) for x in open('data/snp2.csv').readlines()]
    t = time.time()
    series_cutoff = 1000
    import numpy as np
    before = series[0]

    _mean = np.mean(series)
    _sd = np.std(series)
    _min = 20
    _max = 30
    # series = shapelet_utils.normalize(series[:series_cutoff])
    # series = [np.log(x) for x in series[:series_cutoff]]
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # series = scaler.fit_transform(series)
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

    shapelet_utils.graph_classes(final[:k], np.min(series), np.max(series), 20)

    
    file_name = 'shapelets/output/' + '-'.join((str(series_cutoff), str(_min), str(_max), str(mse_threshold), str(len(final)))) + '-standardized.csv'
    with open(file_name, 'w') as f:
        f.write("target,sequence\n")
        for i,shapelet in enumerate(final):
            f.write(str(i) + "," + shapelet.to_csv_standardise() + "\n")
            for similar in shapelet.of_same_class:
                f.write(str(i) + "," + similar.to_csv_standardise() + "\n")


    # shapelet_utils.graph(series[:series_cutoff], final[:k])
    
    # final.sort(key = lambda x: x.quality, reverse=quality_threshold)

    
    



if __name__ == '__main__':  

    parser  = argparse.ArgumentParser()
    parser.add_argument("-g",help="amount of shapelets from each class to display, -g 10")
    parser.add_argument("-s", help="Display series up to cutoff, -s 1000")
    parser.add_argument("-f", help="filename of shapelets to display")

    args = parser.parse_args()

    if args.g:
        if not args.f:
            print ("Please provide a filename to display")

        shapelet_utils.graph_classes_from_file(open(args.f), int(args.g))

        exit()
    # if args.s:


    main()
    exit()

    from trend_lines.simple_lstm import simple_lstm
    from trend_lines.segment import slidingwindowsegment, bottomupsegment, topdownsegment
    from trend_lines.fit import interpolate, sumsquared_error
    from trend_lines.wrappers import stats, convert_to_slope_duration, draw_plot, draw_segments
    mod = simple_lstm()
    mod.train()
    exit(1)
    with open("data/snp2.csv") as f:
    # with open("example_data/16265-normalecg.txt") as f:
        file_lines = f.readlines()

    data = [float(x.split("\t")[0].strip()) for x in file_lines]

    max_error = 500

    #sliding window with simple interpolation
    name = "Sliding window with simple interpolation"
    figure()
    start = time.time()
    segments = slidingwindowsegment(data, interpolate, sumsquared_error, max_error)
    stats(name, max_error, start, segments, data)
    draw_plot(data, name)
    draw_segments(segments)


    #bottom-up with  simple interpolation
    name = "Bottom-up with simple interpolation"
    figure()
    start = time.time()
    segments = bottomupsegment(data, interpolate, sumsquared_error, max_error)
    stats(name, max_error, start, segments, data)
    draw_plot(data,name)
    draw_segments(segments)

    #top-down with  simple interpolation
    name = "Top-down with simple interpolation"
    figure()
    start = time.time()
    segments = topdownsegment(data, interpolate, sumsquared_error, max_error)
    stats(name, max_error, start, segments, data)
    draw_plot(data,name)
    draw_segments(segments)
    
    # only uses from topdown ?
    with open ('slope_dur.csv', 'w') as f:
        for slope, duration in convert_to_slope_duration(segments):
            f.write(  ','.join( ( "%.2f" % slope, "%d" % duration )) + "\n")

    # show()