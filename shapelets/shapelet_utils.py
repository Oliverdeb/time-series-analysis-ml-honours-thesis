from shapelets.shapelet import Shapelet
from scipy.spatial.distance import sqeuclidean

import numpy as np

class shapelet_utils:
    
    @staticmethod
    def percent_diff( series):
        diff = []
        for i in range(1,len(series)):
            diff.append( (series[i] - series [i-1])) 
        return diff

    @staticmethod
    def graph_classes_from_file(in_file, n_classes):
        shapelets = []
        lines = in_file.readlines()
        fst = Shapelet ([float(x) for x in lines[1].split(',')[1].split(' ')], 0)
        fst.of_same_class = set()
        prev_class = '0'
        shapelets.append(fst)
        print ("test")
        for line in lines[2:]:
            line = line.split(',')
            shapelet = line[1].split(' ')
            curr = Shapelet([float(x) for x in shapelet], 0)
            curr.of_same_class = set()
            

            if line[0] == prev_class:
                shapelets[-1].of_same_class.update([curr])
            else:
                prev_class = line[0]
                shapelets[-1].quality = len(shapelets[-1].of_same_class)
                shapelets.append(curr)

        shapelets.sort(key = lambda x: x.quality, reverse=True)
        shapelet_utils.graph_classes(shapelets[:10], -600, 300, n_classes)

    @staticmethod
    def graph_classes(shapelets, _min, _max, n_classes):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, axes = plt.subplots(nrows=1, ncols=len(shapelets))

        for i,shapelet in enumerate(shapelets):
            axes[i].set_title('shapelet' + str(i) + "," + str(len(shapelet.of_same_class) + 1))

            even_y_values = np.linspace(_min, _max, n_classes if len(shapelet.of_same_class) > n_classes else len(shapelet.of_same_class))
            # diff = - ( shapelet.shapelet[0] - even_y_values[0])
            # diff = 0
            axes[i].scatter(range(len(shapelet.shapelet)), shapelet.shapelet)


            for e, (j,similar_shapelet) in zip(even_y_values, enumerate(list(shapelet.of_same_class)[:n_classes])):
                # print (":test")
                diff =  ( similar_shapelet.shapelet[0] - e)

                # axes[i].scatter(range(len(similar_shapelet.shapelet)),  [y - j*35 for y in similar_shapelet.shapelet])
                axes[i].scatter(range(len(similar_shapelet.shapelet)),  similar_shapelet.shapelet + diff )#if j % 2 ==0 else similar_shapelet.shapelet - diff)
            axes[i].set_ylim([_min*0.85 , _max*1.05])
            # axes[i].set_ylim([-3, 3])
        
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
        while(len(shapelets) > 0):
            if len(shapelets[0].of_same_class) < 12:
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
        print ()
        return final

    @staticmethod
    def remove_items_from_other_shapelet_classes(set_of_shapelets_seen, shapelets):
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
                    # if i has a better quality, ie high, set new index
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
                for shapelet in shapelets 
                # if  abs (shapelet.start_index - candidate.start_index) > 10
                if abs (len(candidate.shapelet) - len(shapelet.shapelet)) <= 5
                and shapelet_utils.mse_dist(candidate.shapelet, shapelet.shapelet, threshold) 
        }

    @staticmethod
    def mse_dist(s1, s2, threshold):
        diff = - (s1[0] - s2[0])
        shapelet = s2 + diff if diff != 0 else s2
        shorter = len(s1) if len(s1) < len(s2) else len(s2)
        for i in range(1, shorter):
            if abs(s1[i] - shapelet[i]) > abs(threshold):
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
        return [
            (shapelet_utils.MSE(candidate.shapelet, shapelet.shapelet) , shapelet) 
            for shapelet in shapelets if candidate.start_index != shapelet.start_index
        ]

    @staticmethod
    def standardize(series):
        from scipy.stats import zscore
        return zscore(series)