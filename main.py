from shapelets.shapelet import Shapelet
from shapelets.shapelet_utils import shapelet_utils
from pandas import read_csv
import numpy as np
import time, argparse, multiprocessing, os

def work(pid, out_q, my_pool, pool, mse_threshold):
    """
    Parallel worker processing candidates
    """
    tabs = '\t' * pid
    total = float(len(my_pool))
    
    from random import randint
    prime = [31,37,41,43,47,53,59,61][randint(0, 8 -1)]

    for i,candidate in enumerate(my_pool):
        if pid < 20 and i % prime == 0 :
            print ("\r%s%.2f"%(tabs, i/total), end="")

        # build up the shapelet class with other instances found that are deemed to be of the same class
        candidate.of_same_class = shapelet_utils.find_new_mse(candidate, pool, mse_threshold)
        candidate.quality = len(candidate.of_same_class)

        # add shapelet candidateto the shared queue
        out_q.put(candidate)

    # put 'DONE' on the queue to indicate when finished
    out_q.put('DONE')

def output_to_file(file_name, shapelets, shapelet_dict, mse):
    """
    Pickle the:
        - shapelet dictionary
        - shapelet list
        - mse used for extraction
    to a file so that it can be loaded in later for testing in the prediction phase. Original shapelets are needed
    """

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
                'mse' : mse,
        }, out_map)
    dump_csv(shapelets, shapelet_dict, 49, file_name)
    

def dump_csv(shapelets, shapelet_dict, max_per_class, file_name):
    """
    Function that dumps shapelets to csv file with their labels to train a classifier
    """
    
    with open(file_name, 'w') as f:
        f.write("target,sequence\n")
        for i,shapelet in enumerate(shapelets):
            f.write(str(i) + "," + shapelet.to_csv_offset_0() + "\n")
            for similar in shapelet.of_same_class_objs(shapelet_dict, max_per_class):
                f.write(str(i) + "," + similar.to_csv_offset_0() + "\n")
                
def main(mse=0.5, n_procs=8, min=None, max=None, snp=False):
    """
    Main method for shapelet extraction
    """
    t = time.time()

    _min = int(min) if min else 35
    _max = int(max) if max else 35
    N_PROCS = int(n_procs) if n_procs else 8

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
            # load in and parse all the datasets
            dataset = read_csv(os.path.join(archive_dir, _file), usecols=[1])
            datasets.append((
                [x[0] for x in dataset.values.astype('float32')],
                _file))

    shapelet_dict = {}
    candidate_pool = []

    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(datasets)))

    for (dataset, file_name), color in zip(datasets, colors):
        # Loop through all 35 datasets creating candidates from each
        # build up candidate pool

        print ("Looking at dataset: %s. Shapelet ID starting from: %d" % (file_name, id))
        if file_name in ('StatisticsHistory-REDEFINE-2018-06-04.csv', 'StatisticsHistory-NASPERS-N-2018-06-04.csv'):
            # only use the first 70% of these two datasets, the last 30% will be used for testing in predict.py
            print (file_name, 'ONLY USING 70%' )
            cutoff = int(0.7 * len(dataset))
            dataset = dataset[:cutoff]

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

    # shared queue to be used between processes
    out_q = multiprocessing.Queue()
    lower = 0
    
    for i in range(0, N_PROCS):
        pid = i
        upper = int(np.ceil(lower + n_candidates_per_proc))
        pool_range = (lower, upper)
        print ("process {}: {} candidates from {} of pool".format(pid, upper-lower, pool_range ))
        lower = upper

        # create processes to process a sub part of the candidate pool
        p = multiprocessing.Process(
            target=work,
            args=(pid, out_q, pool[pool_range[0]: pool_range[1]], pool, mse_threshold,)
        )
        procs.append(p)

    print ("Displaying a max of 20 process outputs")

    # start processes
    for p in procs:
        p.start()

    shapelets = []
    procs_done = 0

    # get all processed candidates from the shared queue between processes
    while procs_done != N_PROCS:        
        result = out_q.get()
        if result == 'DONE':
            # continue checking queue until all processes have put 'DONE' onto the queue, indicating they 
            # have finished processing
            procs_done += 1
        else:
            shapelets.append(result)

    # wait for all processes to complete
    for p in procs:
        p.join()

    print ("\nDONE")
    print()
    
    shapelets.sort(key = lambda x: x.quality, reverse=True)

    print ("Time taken to compute all: %.2fs" % (time.time() - t))    
    print ("%d candidates before removing duplicates and removing classes without sufficient candidates" % len(shapelets))
    final = shapelet_utils.remove_duplicates(shapelets, 20)
    print ("%d candidates after" % len(final))
    print ("%.2fs elapsed\n%d initial shapelets found\n%d after class check" % (time.time() -t, len(shapelets), len(final)))

    file_name = 'shapelets/output/std_%d-' % len(pool) + \
     '-'.join((str(_min), str(_max), str(mse_threshold), str(len(final))))+ '.csv'

    output_to_file(file_name, final, shapelet_dict, mse_threshold)

    _min = 0
    _max = 10 + 10*1
    
    # graph classes once extraction is complete
    shapelet_utils.graph_classes(final[:k], k, _min, _max, shapelet_dict)
    
if __name__ == '__main__':  

    # all possible command line args for running trend line or shaplet extraction
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
    parser.add_argument('-cutoff', help='cutoff number of shapelet classes', default=None)
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

        # extract trends and train model with various hyper parameters
        Trendrunner().run(int(args.mse))
        Trendrunner().train_baby_train(int(args.epochs), int(args.batch), args.f, int(args.look_back))
        
    else:
        if args.csv:
            # load in shapelets from graph file and output to csv for training LSTM
            # filter out shapelets that have too few instances and restrict shapelets
            # that have too many instances, to not have class imbalance when training model
            if args.min is None or args.max is None:
                print ("provide min and max per class")
                exit(1)
            min_per_class, max_per_class = int(args.min), int(args.max)
            cutoff = None if args.cutoff is None else int(args.cutoff)
            from pickle import load
            in_dict = load(open(args.csv, 'rb'))
            shapelets = in_dict['shapelets']
            shapelet_dict = in_dict['shapelet_dict']
            shapelets = shapelet_utils.remove_duplicates(shapelets, min_per_class)
            shapelets = shapelets[:cutoff]
            out_file = 'reprocessed%d-%s' % (len(shapelets), args.csv.rpartition('/')[2].replace('.graph','.csv'))
            print ('outfile is:  %s' % out_file)
            print ('new number of classes is %d' % len(shapelets))
            dump_csv(shapelets, shapelet_dict, max_per_class,  out_file)
            exit(0)

        elif args.g:
            # load shapelets in from a file to graph/visualise
            if not args.f:
                print ("Please provide a filename to display")

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
                shapelet_utils.graph_classes2(shapelets[:k], per_class, _min, _max, shapelet_dict)

                # shapelet_utils.graph_classes_shapelets(shapelets[:k], per_class, _min, _max, shapelet_dict)
            ind = 0
            for shape in shapelets[ind:ind+1]:
                
                count = {}
                count[shape.dataset_name] = 1

                for instance in shape.of_same_class_objs(shapelet_dict):
                    if count.get(instance.dataset_name) is None:
                        count[instance.dataset_name] = 1
                    else:
                        count[instance.dataset_name] += 1
            from random import randint
            _file = list(count.keys())[randint(0,len(count)-1)]

            from pandas import read_csv
            archive_dir = 'data/jse'
            path = os.path.join(archive_dir, _file)
            df = read_csv(path, usecols=[1])
            series = df.values.astype('float32')

            # shapes to display
            first = [shapelets[x].of_same_class_objs(shapelet_dict) + [shapelets[x]] for x in range(1)]
            first += [shapelets[x].of_same_class_objs(shapelet_dict) + [shapelets[x]] for x in range(3,5)]
            shapes = [[x for x in _class if x.dataset_name == _file] for _class in first]
            shapelet_utils.graph_shapes_classes_on_series(shapes, series, _file, lb=len(shapelets[0].shapelet))
            exit(0)
        
        if args.mse.find(',') == -1:
            main(args.mse, args.procs, args.min, args.max, args.snp)
        else:
            for mse in (float(x) for x in args.mse.split(',')):
                main(mse, args.procs, args.min, args.max, args.snp)
