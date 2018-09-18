try:
    # incase this is imported from another directory
    from shapelets.shapelet import Shapelet
except ImportError:
    from shapelet import Shapelet
from scipy.spatial.distance import sqeuclidean
from scipy.stats import zscore

import numpy as np


class shapelet_utils:
    """
    Helper class containing various methods related to shapelet extraction and visualisation
    """

    @staticmethod
    def graph_shapes_on_series(shapelets, series, series_name, lb=20, _type='-ok', color='green'):
        """
        Function that graphs a list of shapelets on a time series
        """
        import matplotlib.pyplot as plt
        
        plt.plot(series, label=series_name)
        first = True
        for shape in shapelets:
            x_data = list(range(shape.start_index, shape.start_index + len(shape.shapelet)))
            if first:
                label ='shapelet class'
                pred_label = 'predicted'
                first = False
            else:
                label = None
                pred_label = None
            plt.plot(x_data[:lb], shape.shapelet[:lb], _type, label=label, color=color)
            if lb < len(shape.shapelet):
                plt.plot(x_data[lb-1:], shape.shapelet[lb-1:], _type, label=pred_label, color='red')
        plt.legend(loc='best')
        plt.xlabel("Time (days)")
        plt.ylabel("Price (ZAR)")
        plt.show()

    @staticmethod
    def graph_shapes_classes_on_series(shapelets, series, series_name, lb=20, _type='-k'):
        """
        Function that graphs a list of shapelet classes on a time series
        Each class can contain a number of shapelets
        Shapelets from the same class are displayed as the same color
        """
        import matplotlib.pyplot as plt
        colors = ['magenta', 'red', 'orange', 'purple']        
        plt.plot(series, label=series_name)
        first = True
        for i,(_class,color) in enumerate(zip(shapelets, colors)):
            for shape in _class:
                x_data = list(range(shape.start_index, shape.start_index + len(shape.shapelet)))
                if first:
                    label ='shapelet class %d' % i
                    first = False
                else:
                    label = None
                plt.plot(x_data[:lb], shape.shapelet[:lb], _type, label=label, color=color)
            first = True
        plt.legend(loc='best')
        plt.xlabel("Time (days)")
        plt.ylabel("Price (ZAR)")
        plt.show()

    @staticmethod
    def search_all(future_shape, shapelets, shapelets_dict):
        """
        Function that searches through all shapelets returning the ID of the best match
        """
        min_dist = np.iinfo('int32').max
        min_id = None

        for shape in shapelets:

            sum_dist = shape @ future_shape
            if sum_dist < min_dist:
                min_dist = sum_dist
                min_id = shape.id

            for instance in shape.of_same_class_objs(shapelets_dict, 99):
                sum_dist = instance @ future_shape
                if sum_dist < min_dist:
                    min_dist = sum_dist
                    min_id = instance.id
        return min_id

    @staticmethod
    def search_classes(future_shape, shapelets, shapelets_dict, threshold):
        """
        Function that searches 'horizontally' through shapelet classes 
        for the best class label matched
        """
        min_dist = np.iinfo('int32').max
        min_label = None
        for label,shape in enumerate(shapelets):
            sum_dist = shape @ future_shape
            if sum_dist < min_dist:
                min_dist = sum_dist
                min_label = label
        return min_label

    @staticmethod
    def search_instance_of_class(future_shape, class_id, shapelet_dict, threshold):
        """
        Function that searches 'vertically' through a particular shapelet class
        for the best matched instance of that class
        """
        min = np.iinfo('int32').max
        min_id = None

        of_same_class = shapelet_dict[class_id].of_same_class_objs(shapelet_dict)
        
        for instance in of_same_class + [shapelet_dict[class_id]]:
            sum_dist = future_shape @ instance
            if sum_dist < min:
                min = sum_dist
                min_id = instance.id
        return min_id

    @staticmethod
    def graph_classes(shapelets, per_class, _min, _max, shapelet_dict):
        """
        Function that plots shapelet classes and a number of instances per class
        """
        import matplotlib.pyplot as plt
        n_classes = per_class
        fig, axes = plt.subplots(nrows=1, ncols=len(shapelets))
        for i, shapelet in enumerate(shapelets):
            axes[i].set_title('class: %s' %(str(i)))
            even_y_values = np.linspace(_min, _max, n_classes + 1)
            y_vals = shapelet.std_shapelet - \
                shapelet.std_shapelet[0] + even_y_values[0]
            axes[i].get_xaxis().set_ticks([])
            axes[i].get_yaxis().set_ticks([])
            axes[i].plot(range(len(shapelet.shapelet)),
                            y_vals, '-k', c=shapelet.color)
            
            j = 1
            seen_sets = set()
            for similar_shapelet in shapelet.of_same_class_objs(shapelet_dict):
                if similar_shapelet.dataset_name not in seen_sets:
                    e = even_y_values[j]
                    seen_sets = seen_sets | {similar_shapelet.dataset_name}
                    j += 1
                    y_vals = similar_shapelet.std_shapelet - \
                        similar_shapelet.std_shapelet[0] + e
                    axes[i].plot(range(len(similar_shapelet.shapelet)),
                                    y_vals, '-k', c=similar_shapelet.color)
                if j == n_classes:
                    break
                
        plt.xlabel("Time (days)")
        plt.ylabel("Price (ZAR)")
        plt.show()
    
    @staticmethod
    def graph_classes2(shapelets, per_class, _min, _max, shapelet_dict):
        """
        Function that plots shapelet classes and a number of instances per class
        """
        import matplotlib.pyplot as plt
        n_classes = per_class
        fig, axes = plt.subplots(nrows=1, ncols=len(shapelets))
        for i, shapelet in enumerate(shapelets):
            axes[i].set_title('class: %s' %(str(i)))
            even_y_values = np.linspace(_min, _max, n_classes + 1)
            y_vals = shapelet.shapelet - \
                shapelet.shapelet[0] + even_y_values[0]
            axes[i].get_xaxis().set_ticks([])
            axes[i].get_yaxis().set_ticks([])
            axes[i].plot(range(len(shapelet.shapelet)),
                            y_vals, '-k', c=shapelet.color)
            
            j = 1
            seen_sets = set()
            for similar_shapelet in shapelet.of_same_class_objs(shapelet_dict):
                if similar_shapelet.dataset_name not in seen_sets:
                    e = even_y_values[j]
                    seen_sets = seen_sets | {similar_shapelet.dataset_name}
                    j += 1
                    y_vals = similar_shapelet.shapelet - \
                        similar_shapelet.shapelet[0] + e
                    axes[i].plot(range(len(similar_shapelet.shapelet)),
                                    y_vals, '-k', c=similar_shapelet.color)
                if j == n_classes:
                    break
                
        plt.xlabel("Time (days)")
        plt.ylabel("Price (ZAR)")
        plt.show()

    @staticmethod
    def graph_classes_shapelets(shapelets, per_class, _min, _max, shapelet_dict):
        """
        Function that plots shapelet classes and a number of instances per class
        """
        import matplotlib.pyplot as plt
        n_classes = per_class
        fig, axes = plt.subplots(nrows=1, ncols=len(shapelets))

        for i, shapelet in enumerate(shapelets):
            axes[i].set_title('shapelet' + str(i) + "," +
                              str(len(shapelet.of_same_class) + 1))
            even_y_values = np.linspace(_min, _max, n_classes + 1)
            shapelet.shapelet = shapelet.shapelet - \
                shapelet.shapelet[0] + even_y_values[0]

            axes[i].plot(range(len(shapelet.shapelet)),
                            shapelet.shapelet, c=shapelet.color)
            for e, similar_shapelet in zip(even_y_values[1:], shapelet.of_same_class_objs(shapelet_dict, n_classes)):
                similar_shapelet.shapelet = similar_shapelet.shapelet - \
                    similar_shapelet.shapelet[0] + e
                axes[i].plot(range(len(similar_shapelet.shapelet)),
                                similar_shapelet.shapelet, c=similar_shapelet.color)
        fig.tight_layout()
        plt.xlabel("Time (days)")
        plt.ylabel("Price (ZAR)")
        plt.show()

    @staticmethod
    def remove_duplicates(shapelets, min_per_class):
        """
        Function that removes duplicate shapelets from other classes
        """
        shapelets.sort(key=lambda x: x.quality, reverse=True)
        final = []
        set_of_shapelets_seen = set()
        i = 0

        while(len(shapelets) > 0):
            if len(shapelets[0].of_same_class) <= min_per_class:
                break
            curr_shapelet = shapelets[0]
            final.append(curr_shapelet)
            set_of_shapelets_seen.update(
                [curr_shapelet.id], curr_shapelet.of_same_class)
            del shapelets[0]
            shapelet_utils.remove_items_from_other_shapelet_classes(
                set_of_shapelets_seen, shapelets)

            shapelets.sort(key=lambda x: x.quality, reverse=True)
            i += 1

        return final

    @staticmethod
    def remove_items_from_other_shapelet_classes(set_of_shapelets_seen, shapelets):
        """
        Function that ensures shapelets do not contain duplicates
        """
        for shapelet in shapelets:
            shapelet.of_same_class = shapelet.of_same_class - set_of_shapelets_seen
            shapelet.quality = len(shapelet.of_same_class)

    @staticmethod
    def merge(k_shapelets, shapelets):
        """
        Function that merges a list of shapelets into the main list, without needing to resort
        """
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
    def generate_all_size_candidates(dataset, name, id, _min, _max, color):
        """
        Generate all shapelet candidates, between a minimum and maximum size
        """
        candidates = []
        shapelet_dict = {}

        for l in range(_min, _max):
            for i in range(len(dataset) - l + 1):

                shapelet = Shapelet(
                    shapelet=np.array(dataset[i:i+l]),
                    index=i,
                    dataset_name=name,
                    color=color,
                    id=id
                )

                if shapelet.std == 0:
                    # if candidate has stddev of 0, means all values are the same
                    # not a useful shape, causes problems, cant standardize.
                    print('ignoring')
                    continue

                candidates.append(shapelet)
                shapelet_dict[id] = shapelet
                id += 1
        return shapelet_dict, candidates, id

  
    @staticmethod
    def find_new_mse(candidate, shapelets, threshold):
        """
        Returns a list of candidate shapelets that are deemed to be of the same class as
        the original shape
        """
        candidates = set()
        candidate_length = len(candidate.shapelet)
        
        for shapelet in shapelets:
            if shapelet.dataset_name == candidate.dataset_name and abs (shapelet.start_index - candidate.start_index) <= 10:
                continue
            
            if candidate.id != shapelet.id \
            and abs(candidate_length - len(shapelet.shapelet)) <= 5 \
            and shapelet_utils.mse_dist(candidate.std_shapelet, shapelet.std_shapelet, threshold):
                candidates.add(shapelet.id)

        return candidates           
        
    @staticmethod
    def std_mse_dist(fst, snd, threshold):
        """
        Returns true/false if two shapelets (standardized) are below the threshold value
        standardizes them up to the length of the shorter shapelet to remove bias between shapes
        of different lengths
        """
        shorter = len(fst.std_shapelet) if \
            len(fst.std_shapelet) < len(snd.std_shapelet) \
                else len(snd.std_shapelet)
                
        if fst.shapelet is None:
            fst = fst.std_shapelet[:shorter]
            snd = zscore(snd.shapelet[:shorter])
        elif snd.shapelet is None:
            fst = zscore(fst.shapelet[:shorter])
            snd = snd.std_shapelet[:shorter]
        for i in range(0, shorter):
            if abs(fst[i] - snd[i]) > abs(threshold):
                return False
        return True

    @staticmethod
    def mse_dist(fst, snd, threshold):
        """
        Returns true or false if the two shapelets are below the threshold  value
        """
        shapelet = snd
        shorter = len(fst) if len(fst) < len(snd) else len(snd)

        for i in range(0, shorter):
            if abs(fst[i] - shapelet[i]) > abs(threshold):
                return False
        return True