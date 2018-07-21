from shapelets.shapelet import Shapelet
from scipy.spatial.distance import euclidean

import numpy as np

class shapelet_utils:
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
        return shapelet_utils.remove_similar(shapelets, threshold)

    @staticmethod
    def merge(k_shapelets, shapelets):
        # k_shapelets.sort(key = lambda x: x.quality)

        k_i = 0
        s_i = 0
        merged = []
        while k_i < len(k_shapelets) and s_i < len(shapelets):
            if k_shapelets[k_i] < shapelets[s_i]:
                merged.append(k_shapelets[k_i])
                k_i += 1
            else:
                merged.append(shapelets[s_i])
                s_i += 1
        if k_i == len(k_shapelets):
            merged += shapelets[s_i:]
        else:
            merged += k_shapelets[k_i:]
        return merged

    @staticmethod
    def generate_candidates(dataset, window_size):
        candidates = []
        for i in range(len(dataset) - window_size + 1):
            candidates.append(Shapelet(dataset[i:i+window_size], i))
        return candidates

    # TODO: improve to use "sufficient statistics", as per Logical shapelets and "A discriminative shapelets transformation for time series classification"
    @staticmethod
    def subsequence_distance(fst, snd):

        diff = snd[0] - fst[0]
        diff = - diff if diff < 0 else diff   
        
        shapelet = [x + diff for x in snd] if diff != 0 else snd

        dist = euclidean(shapelet, fst) 
        return dist / len(snd)

    @staticmethod
    def find_mse(fst, snd):
        return [shapelet_utils.subsequence_distance(fst.shapelet, s.shapelet) for s in snd]

    @staticmethod
    def normalize(series):
        from scipy.stats import zscore
        return zscore(series)

    @staticmethod
    def distance(shapelet, series): 
        return euclidean(shapelet, series)

    def generate_shapelets(self):
        pass