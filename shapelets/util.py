from shapelets.shapelet import Shapelet

import numpy as np

class util:
    def __init__(self):
        pass

    @staticmethod
    def graph(series, shapelets):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(len(series)), series, c='b', marker='s', label='series')

        colors = cm.rainbow(np.linspace(0, 1, len(shapelets)))

        for (i,shapelet), c in zip(enumerate(shapelets), colors):
            index = shapelet.start_index
            to_plot = [x for x in shapelet.shapelet]
            ax.scatter(range(index, index+len(shapelet.shapelet)), to_plot, c=c, marker='o', label='shapelet'+str(i))

        
        plt.legend(loc='best')

        plt.show()


    @staticmethod
    def remove_similar(shapelets, threshold):
        new_s = []
        index = 0
        for i in range(1, len(shapelets)):

            if shapelets[i].start_index - shapelets[index].start_index < threshold:
                if shapelets[index] > shapelets[i]:
                    # if i has a better quality, ie lower, set new index
                    index = i
            else:
                new_s.append(shapelets[index])
                index = i
        return new_s

    @staticmethod
    def remove_all_similar(shapelets, threshold):
        # from copy import deepcopy
        # shapes = deepcopy(shapelets)
        shapelets.sort(key= lambda x: x.start_index)
        return util.remove_similar(shapelets, threshold)

    @staticmethod
    def merge(k, k_shapelets, shapelets):
        k_shapelets.sort(key = lambda x: x.quality)

        k_index = 0
        for i in range (len (shapelets)):
            if k_index == len(k_shapelets):
                break
            if k_shapelets[k_index] < shapelets[i]:
                shapelets.insert(i, k_shapelets[k_index])
                k_index += 1
        if k_index < len(k_shapelets):
            shapelets =  shapelets + k_shapelets[k_index:]
        return shapelets

    @staticmethod
    def generate_candidates(dataset, _min, _max):
        candidates = []
        for l in range(_min, _max):
            for i in range(len(dataset) - l + 1):
                candidates.append(Shapelet(dataset[i:i+l], i))
        return candidates

    # TODO: improve to use "sufficient statistics", as per Logical shapelets and "A discriminative shapelets transformation for time series classification"
    @staticmethod
    def subsequence_distance(series, shapelet):
        from scipy.spatial.distance import euclidean

        l = len(shapelet)
        s = len(series)

        diff = shapelet[0] - series[0]
        if diff < 0:
            shapelet = [x + -diff for x in shapelet]
        else:
            shapelet = [x - diff for x in shapelet]

        # make min = max_int, index = 0, slice = []
        (min_dist, min_dist_index, _slice) = (np.iinfo(np.int32)).max, 0, []

        # for 0 up to length of series - length of shapelet + 1
        for i in range(s - l + 1):
            current_slice = series[i:i+l] # current window
            dist = euclidean(shapelet, current_slice) # dist between shapelet and window
            if dist < min_dist:
                # update best match so far
                (min_dist, min_dist_index, _slice) = dist, i, current_slice
        return min_dist

    @staticmethod
    def find_mse(candidate, shapelets):
        return [util.subsequence_distance(shapelet.shapelet, candidate.shapelet) for shapelet in shapelets]

    @staticmethod
    def normalize(series):
        from scipy.stats import zscore
        return zscore(series)

    @staticmethod
    def distance(shapelet, series): 
        return euclidean(shapelet, series)

    def generate_shapelets(self):
        pass