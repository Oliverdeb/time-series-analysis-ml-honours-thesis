from shapelets.shapelet import Shapelet
from shapelets.shapelet_utils import shapelet_utils
# from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
# from shapelets.classifier import LSTMClassifier
import numpy as np
import time, argparse, multiprocessing

def work(pid, out_q, my_pool, pool, mse_threshold):
    tabs = '\t' * pid
    total = len(my_pool)
    results = []
    for i,candidate in enumerate(my_pool):
        if i % 23 ==0 :
            print ("\r%s%.2f"%(tabs, i/total), end="")

        candidate.of_same_class = shapelet_utils.find_new_mse(candidate, pool, mse_threshold)
        candidate.quality = len(candidate.of_same_class)
        out_q.put(candidate)
    out_q.put('DONE')
    # with open(str(pid) + '.out', 'w') as f:
    #     pickle.dump(results, f)

def output_to_file(file_name, shapelets, shapelet_dict, series_min, series_max):

    output_map = {}
    
    for shape in shapelets:
        output_map.update({shape.id : shape})
        for id in shape.of_same_class:
            output_map.update({id : shapelet_dict[id]})
    
    from pickle import dump

    with open(file_name.replace('.csv', '.graph'), 'wb') as out_map:
        dump({ 
                'shapelet_dict': output_map,
                'shapelets': shapelets,
                'min': series_min,
                'max': series_max,
        }, out_map)

    with open(file_name, 'w') as f:
        f.write("target,sequence\n")
        for i,shapelet in enumerate(shapelets):
            f.write(str(i) + "," + shapelet.to_csv() + "\n")
            for similar in shapelet.of_same_class_objs(shapelet_dict):
                f.write(str(i) + "," + similar.to_csv() + "\n")

def main():
    series = [float(x) for x in open('data/snp2.csv').readlines()]
    t = time.time()
    series_cutoff = 3416
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
    mse_threshold = 0.5
    print ("mse %.4f, mean %.4f, sd %.4f" % (mse_threshold, _mean, _sd))
    print ("before is %.2f, after is %.2f" % (before, series[0]))
    # mse_threshold = mse_threshold / (before / series[0])
    # mse_threshold = 0.5
    print ("new mse %.4f" % mse_threshold)
    
    

    for dataset in sets:
        
        shapelets = []
        shapelet_dict, candidates_i_l = shapelet_utils.generate_all_size_candidates(dataset, _min, _max + 1)

        # convert to numpy array
        pool = np.array(candidates_i_l)

        # shuffle array so that work is evenly distributed betweens processes
        # np.random.shuffle(pool)s

        N_PROCS = 8

        # print (sys.getrecursionlimit())
        print ("{0} candidates, {1} processes spawning".format(len(pool), N_PROCS))

        procs = []
        n_candidates_per_proc = len(pool) // N_PROCS

        out_q = multiprocessing.Queue()
        # proc_pool = multiprocessing.Pool(processes=N_PROCS)        

        for i_proc in range(0, N_PROCS*n_candidates_per_proc, n_candidates_per_proc):
            pid = i_proc // n_candidates_per_proc
            pool_range = (i_proc, i_proc + n_candidates_per_proc)
            print ("process {}: {} candidates from {} of pool".format(pid, n_candidates_per_proc, pool_range ))
            # print (pid, pool[pool_range[0]: pool_range[1]][:2], pool[:2], mse_threshold)
            p = multiprocessing.Process(
                target=work,
                args=(pid, out_q, pool[pool_range[0]: pool_range[1]], pool, mse_threshold,)
            )

            procs.append(p)
            # procs.append(proc_pool.apply_async(work, args=(pid, pool[pool_range[0]: pool_range[1]], pool, n_candidates_per_proc, mse_threshold,)))
        for p in procs:
            p.start()

        results = []
        procs_done = 0

        while procs_done != N_PROCS:
            result = out_q.get()
            if result == 'DONE':
                procs_done += 1
            else:
                results.append(result)

        for p in procs:
            p.join()

        print ("DONE")
            
        shapelets = results
        print()
        
        # shapelets = shapelet_utils.remove_all_similar(shapelets, (_min + _max) / 5.0)
        shapelets.sort(key = lambda x: x.quality, reverse=True)
        # k_shapelets = shapelet_utils.merge(k_shapelets, shapelets)
        k_shapelets = shapelets

        

    print ("time taken to get all %.2f" % (time.time() - t))    
   
    final = shapelet_utils.remove_duplicates(k_shapelets)
    
    print ("%.2fs elapsed\n%d initial shapelets found\n%d after class check" % (time.time() -t, len(k_shapelets), len(final)))

    file_name = 'shapelets/output/std' + '-'.join((str(series_cutoff), str(_min), str(_max), str(mse_threshold), str(len(final)))) + '.csv'

    output_to_file(file_name, final, shapelet_dict, np.min(series), np.max(series))

    shapelet_utils.graph_classes(final[:k], k, np.min(series), np.max(series), shapelet_dict)
        
    # shapelet_utils.graph(series[:series_cutoff], final[:k])
    
    # final.sort(key = lambda x: x.quality, reverse=quality_threshold)
    
if __name__ == '__main__':  

    parser  = argparse.ArgumentParser()
    parser.add_argument("-g",help="amount of shapelets from each class to display, -g 10")
    parser.add_argument("-p",help="eggs per class, -p 10")
    parser.add_argument("-s", help="Display series up to cutoff, -s 1000")
    parser.add_argument("-f", help="filename of shapelets to display")
    parser.add_argument("-train", help="filename of shapelets to display", action='store_true')

    args = parser.parse_args()


    if args.g:
        if not args.f:
            print ("Please provide a filename to display")

        # shapelet_utils.graph_classes_from_file(open(args.f), int(args.g))

        from pickle import load

        in_dict = load(open(args.f, 'rb'))
        shapelets = in_dict['shapelets']
        shapelet_dict = in_dict['shapelet_dict']
        _min = in_dict['min']
        k = int (args.g)
        per_class = int (args.p)
        _max = 400 + per_class*75

        
        shapelet_utils.graph_classes(shapelets[:k], per_class, _min, _max, shapelet_dict)
        exit()

    file_name = main()

    # if args.train:
    #     file_name = file_name if not args.f else args.f

    #     lstm = LSTMClassifier(file_name)
    
    # if args.train:

    exit()

    from trend_lines.simple_lstm import simple_lstm
    from trend_lines.segment import slidingwindowsegment, bottomupsegment, topdownsegment
    from trend_lines.fit import interpolate, sumsquared_error
    from trend_lines.wrappers import stats, convert_to_slope_duration, draw_plot, draw_segments
    # mod = simple_lstm()
    # mod.train()
    # exit(1)
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
        f.write('slope,duration')
        for slope, duration in convert_to_slope_duration(segments):
            f.write(  ','.join( ( "%.2f" % slope, "%d" % duration )) + "\n")

    # show()