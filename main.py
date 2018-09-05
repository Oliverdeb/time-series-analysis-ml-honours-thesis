from shapelets.shapelet import Shapelet
from shapelets.shapelet_utils import shapelet_utils
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
# from matplotlib.lines import Line2D
from pandas import read_csv
# from shapelets.classifier import LSTMClassifier
import numpy as np
import time, argparse, multiprocessing, os

def work(pid, out_q, my_pool, pool, mse_threshold):
    tabs = '\t' * pid
    total = float(len(my_pool))
    
    from random import randint
    prime = [31,37,41,43,47,53,59,61][randint(0, 8 -1)]

    for i,candidate in enumerate(my_pool):
        if pid < 20 and i % prime == 0 :
            print ("\r%s%.2f"%(tabs, i/total), end="")

        candidate.of_same_class = shapelet_utils.find_new_mse(candidate, pool, mse_threshold)
        candidate.quality = len(candidate.of_same_class)
        # results.append(candidate)
        out_q.put(candidate)
    # out_q.put(results)
    out_q.put('DONE')

def output_to_file(file_name, shapelets, shapelet_dict):

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
        }, out_map)
    dump_csv(shapelets, shapelet_dict, 49, file_name)
    

def dump_csv(shapelets, shapelet_dict, max_per_class, file_name):
    
    with open(file_name, 'w') as f:
        f.write("target,sequence\n")
        for i,shapelet in enumerate(shapelets):
            f.write(str(i) + "," + shapelet.to_csv_offset_0() + "\n")
            for similar in shapelet.of_same_class_objs(shapelet_dict, max_per_class):
                f.write(str(i) + "," + similar.to_csv_offset_0() + "\n")
                
def main(mse=0.5, n_procs=8, min=None, max=None, snp=False):
    t = time.time()

    _min = int(min) if min else 35
    _max = int(max) if max else 35
    N_PROCS = int(n_procs) if n_procs else 8

    k_shapelets = []
    k = 10
    mse_threshold = float(mse) if mse else 0.5
    
    print ("Using shapelet threshold-cutoff of: %.2f" % mse_threshold)
    print ("Min: %d, Max: %d" % (_min, _max))
    
    id = 0
    datasets = []
    archive_dir = 'data/jse'
    try:
        files = os.listdir(archive_dir)
    except Exception as e:
        print (e)
        print ("Error finding data, is './data/jse' present?")
        exit(1)

    if files == []:
        print ("Error finding datasets, are there any files in './data/jse/*' ?")
        exit(1)

    if args.snp:
        dataset = read_csv('data/snp2.csv', usecols=[0])
        datasets.append(([
            x[0] for x in dataset.values.astype('float32')[:1000]],
            'snp2.csv'))
    else:
        for _file in files:
            dataset = read_csv(os.path.join(archive_dir, _file), usecols=[1])
            datasets.append(([
                x[0] for x in dataset.values.astype('float32')],
                _file))

    shapelet_dict = {}
    candidate_pool = []

    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(datasets)))

    for (dataset, file_name), color in zip(datasets, colors):
        print ("Looking at dataset: %s. Shapelet ID starting from: %d" % (file_name, id))

        shapelets = []
        this_dict, this_pool, id = shapelet_utils.generate_all_size_candidates(dataset, file_name, id, _min, _max + 1, color)
        shapelet_dict.update(this_dict)
        candidate_pool += this_pool
        

    # convert to numpy array
    pool = np.array(candidate_pool)

    # shuffle array so that work is evenly distributed betweens processes
    # np.random.shuffle(pool)

    print ("{0} candidates, {1} processes spawning".format(len(pool), N_PROCS))

    procs = []
    n_candidates_per_proc = len(pool) / N_PROCS

    out_q = multiprocessing.Queue()
    lower = 0
    
    for i in range(0, N_PROCS):
        pid = i
        upper = int(np.ceil(lower + n_candidates_per_proc))
        pool_range = (lower, upper)
        print ("process {}: {} candidates from {} of pool".format(pid, upper-lower, pool_range ))
        lower = upper

        p = multiprocessing.Process(
            target=work,
            args=(pid, out_q, pool[pool_range[0]: pool_range[1]], pool, mse_threshold,)
        )
        procs.append(p)
    print ("Displaying a max of 20 process outputs")
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

    print ("\nDONE")
        
    # shapelets = [shape for result in results for shape in result]
    shapelets = results
    print()
    
    # shapelets = shapelet_utils.remove_all_similar(shapelets, (_min + _max) / 5.0)
    shapelets.sort(key = lambda x: x.quality, reverse=True)
    # k_shapelets = shapelet_utils.merge(k_shapelets, shapelets)
    k_shapelets = shapelets


    print ("Time taken to compute all: %.2fs" % (time.time() - t))    
    print ("%d candidates before removing duplicates and removing classes without sufficient candidates" % len(k_shapelets))
    final = shapelet_utils.remove_duplicates(k_shapelets, 20)
    print ("%d candidates after" % len(final))
    print ("%.2fs elapsed\n%d initial shapelets found\n%d after class check" % (time.time() -t, len(k_shapelets), len(final)))

    file_name = 'shapelets/output/std_%d-' % len(pool) + '-'.join((str(_min), str(_max), str(mse_threshold), str(len(final))))+ '.csv'

    output_to_file(file_name, final, shapelet_dict)

    _min = 0
    _max = 10 + 10*1
    shapelet_utils.graph_classes(final[:k], k, _min, _max, shapelet_dict)
        
    # shapelet_utils.graph(series[:series_cutoff], final[:k])
    
    # final.sort(key = lambda x: x.quality, reverse=quality_threshold)
    
if __name__ == '__main__':  

    parser  = argparse.ArgumentParser()
    parser.add_argument("-g",help="amount of shapelets from each class to display, -g 10", default=None)
    parser.add_argument("-p",help="instances per class, -p 10", default=None)
    parser.add_argument("-s", help="Display series up to cutoff, -s 1000", default=None)
    parser.add_argument("-f", help="filename", default=None)
    parser.add_argument("-shapelets", help="boolean flag for shapelets", action='store_true')
    parser.add_argument("-trends", help="boolean flag for trend lines", action='store_true')
    parser.add_argument('-csv', help='to re process and outputs shapelet to file', default=None)
    parser.add_argument('-min', help='min instances per class before removing', default=None)
    parser.add_argument('-mse', help='mse for trend error', default=None)
    parser.add_argument('-epochs', help='number of epochs', default=None)
    parser.add_argument('-procs', help='number of processes to spawn for shapelet extraction', default=None)
    parser.add_argument('-look_back', help='look_back', default=None)
    parser.add_argument('-batch', help='batch size', default=None)
    parser.add_argument('-snp', help='run on snp dataset', action='store_true')
    parser.add_argument('-max', help='max instances per class to include', default=None)
    parser.add_argument('-std', help='display standardized shapelets or not',  action='store_true')

    args = parser.parse_args()

    if args.trends:
        # run trend line stuff
        from trend_lines.trendrunner import Trendrunner
        if args.f is None:
            print ("Please enter file name")
            exit(1)
        if None in (args.epochs, args.batch):
            print ("Enter batch size and epochs")
            exit(1)
        if None in (args.mse, args.look_back):
            print ("Please enter MSE and look_back for trendline extraction")
            exit(1)

        Trendrunner().run(int(args.mse))
        # Trendrunner().train_baby_train(int(args.epochs), int(args.batch), args.f, int(args.look_back))
        
    else:
        if args.csv:
            if args.min is None or args.max is None:
                print ("provide min and max per class")
                exit(1)
            min_per_class, max_per_class = int(args.min), int(args.max)
            
            from pickle import load
            in_dict = load(open(args.csv, 'rb'))
            shapelets = in_dict['shapelets']
            shapelet_dict = in_dict['shapelet_dict']
            shapelets = shapelet_utils.remove_duplicates(shapelets, min_per_class)
            from random import randint
            r = randint(0,10)
            out_file = 'reprocessed%d-%s' % (r, args.csv.rpartition('/')[2].replace('.graph','.csv'))
            print ('outfile is:%s' % out_file)
            dump_csv(shapelets, shapelet_dict, max_per_class,  out_file)
            exit(0)
        elif args.g:
            if not args.f:
                print ("Please provide a filename to display")
            # shapelet_utils.graph_classes_from_file(open(args.f), int(args.g))

            from pickle import load

            in_dict = load(open(args.f, 'rb'))
            shapelets = in_dict['shapelets']
            shapelet_dict = in_dict['shapelet_dict']
            k = int (args.g)
            per_class = int (args.p)
            _min = 0

            if args.std:
                _max = 10 + 10*2
                shapelet_utils.graph_classes(shapelets[:k], per_class, _min, _max, shapelet_dict)
            else:
                _max = 500 + 10*25
                shapelet_utils.graph_classes_shapelets(shapelets[:k], per_class, _min, _max, shapelet_dict)
                
            exit(0)
        
                
        main(args.mse, args.procs, args.min, args.max, args.snp)
