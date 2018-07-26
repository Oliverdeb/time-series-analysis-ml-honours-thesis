from shapelets.shapelet import Shapelet
from scipy.spatial.distance import sqeuclidean

import numpy as np

class shapelet_utils:
    def __init__(self):
        pass

    @staticmethod
    def percent_diff( series):
        diff = []
        for i in range(1,len(series)):
            diff.append( (series[i] - series [i-1])) 
        return diff

    @staticmethod
    def graph_classes(shapelets, series):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        _m = np.max(series)
        fig, axes = plt.subplots(nrows=1, ncols=len(shapelets))

        for i,shapelet in enumerate(shapelets):
            axes[i].set_title('shapelet' + str(i))
            even_y_values = np.linspace(0, _m, len(shapelet.of_same_class) + 1)
            diff = - ( shapelet.shapelet[0] - even_y_values[0])
            # diff = 0
            axes[i].scatter(range(len(shapelet.shapelet)), shapelet.shapelet + diff if diff != 0 else shapelet.shapelet)


            for j,similar_shapelet in enumerate(shapelet.of_same_class):
                # diff = - ( shapelet.shapelet[0] - even_y_values[0])

                axes[i].scatter(range(len(similar_shapelet.shapelet)),  [y - (j*35)for y in similar_shapelet.shapelet])
            axes[i].set_ylim([1300, 2600])
        
        fig.tight_layout()
        # plt.ylim(2000, 2400)
        plt.show()

    @staticmethod
    def remove_duplicates(shapelets):
        shapelets.sort(key = lambda x: x.quality, reverse=True)
        print ("l;en is", len(shapelets))
        final = []
        set_of_shapelets_seen = set()
        i = 0
        n_candidates = len(shapelets)
        while(len(shapelets)>0):
            if len(shapelets[0].of_same_class) < 6:
                break
            if i % 13 == 0:
                print ("\rlen/n=%.2f, using i/n = %.2f" % (1- (len(shapelets) / n_candidates), i/n_candidates), end="")
            curr_shapelet = shapelets[0]
            final.append(curr_shapelet)
            set_of_shapelets_seen.update ([curr_shapelet], curr_shapelet.of_same_class)
            del shapelets[0]
            shapelet_utils.remove_items_from_other_shapelet_classes(set_of_shapelets_seen, shapelets)

            shapelets.sort(key = lambda x: x.quality, reverse=True)
            i += 1
        return final

    @staticmethod
    def remove_items_from_other_shapelet_classes(set_of_shapelets_seen, shapelets):
        # loop through all the remaining shapelets and remove elents from the shapelets_of_same_class
        # for seen in set_of_shapelets_seen:
        #     for shapelet in shapelets:
        #         if seen in shapelet.of_same_class:
        #             shapelet.of_same_class.remove(seen)
        #             shapelet.quality = len(shapelet.of_same_class)

        for shapelet in shapelets:
            shapelet.of_same_class = shapelet.of_same_class - set_of_shapelets_seen
            shapelet.quality = len(shapelet.of_same_class)
    @staticmethod
    def graph(series, shapelets):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(len(series)), series, c='b', marker='s', label='series')

        colors = cm.rainbow(np.linspace(0, 1, len(shapelets)))

        # for (i,shapelet), c in zip(enumerate(shapelets), colors):
        #     index = shapelet.start_index
        #     to_plot = [y + (i+1) * 4 for y in shapelet.shapelet]
        #     ax.scatter(range(index, index+len(shapelet.shapelet)), to_plot, c=c, marker='o', label='shapelet'+str(i))   
        #     for j,similar_shapelet in enumerate(shapelet.of_same_class):
        #         index = similar_shapelet.start_index
        #         to_plot = [y + (i+2) * 4 + j  for y in similar_shapelet.shapelet]
        #         ax.scatter(range(index, index+len(similar_shapelet.shapelet)), to_plot, c=c, marker='o', label='shapelet'+str(i))        
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
    def generate_all_size_candidates(dataset, _min, _max):
        candidates = []
        for l in range(_min, _max):
            for i in range(len(dataset) - l + 1):
                candidates.append(Shapelet(np.array(dataset[i:i+l]), i))
        return candidates

    @staticmethod
    def generate_candidates(dataset, window_size):
        candidates = []
        for i in range(len(dataset) - window_size + 1):
            candidates.append(Shapelet(np.array(dataset[i:i+window_size]), i))
        return candidates

    # TODO: improve to use "sufficient statistics", as per Logical shapelets and "A discriminative shapelets transformation for time series classification"

    @staticmethod
    def find_new_mse(candidate, shapelets, threshold):
        return {
            shapelet
         for shapelet in shapelets if shapelet.start_index != candidate.start_index and shapelet_utils.mse_dist(candidate.shapelet, shapelet.shapelet, threshold) }

    @staticmethod
    def mse_dist(s1, s2, threshold):
        
        abs_diff = np.abs(s1[:len(s2)] - s2) if len(s1) > len(s2) else np.abs(s1 - s2[:len(s1)])

        for elem in abs_diff:
            if elem > threshold:
                return False
        return True

     

    @staticmethod
    def MSE(fst, snd):        
        diff = - (snd[0] - fst[0])
        shapelet = snd + diff if diff != 0 else snd
        dist = sqeuclidean(shapelet, fst) 
        return dist / len(snd)

    @staticmethod
    def find_mse(candidate, shapelets):
        return [(
            shapelet_utils.MSE(candidate.shapelet, shapelet.shapelet) , shapelet
        ) for shapelet in shapelets if candidate.start_index != shapelet.start_index]

    @staticmethod
    def normalize(series):
        from scipy.stats import zscore
        return zscore(series)

    @staticmethod
    def distance(shapelet, series): 
        return euclidean(shapelet, series)

    def generate_shapelets(self):
        pass