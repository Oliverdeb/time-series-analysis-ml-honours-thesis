from scipy.spatial.distance import euclidean
import numpy as np

class Generation:

    def __init__(self, series):
        self.mean = np.mean(series)
        self.var = np.var(series)

    # TODO: improve to use "sufficient statistics", as per Logical shapelets and "A discriminative shapelets transformation for time series classification"
    @staticmethod
    def subsequence_distance(series, shapelet):
        l = len(shapelet)
        s = len(series)

        # make min = max_int, index = 0, slice = []
        (min_dist, min_dist_index, _slice) = (np.iinfo(np.int32)).max, 0, []

        # for 0 up to length of series - length of shapelet + 1
        for i in range(s - l + 1):
            current_slice = series[i:i+l] # current window
            dist = euclidean(shapelet, current_slice) # dist between shapelet and window
            if dist < min_dist:
                # update best match so far
                (min_dist, min_dist_index, _slice) = dist, i, current_slice

        return (min_dist, min_dist_index, _slice)

                

    @staticmethod
    def normalize():
        sd = np.sqrt(self.var)
        self.series_normalized = [ ( x - self.mean) / self.sd for x in self.series ]

     @staticmethod
    def distance(shapelet, series): 
        return euclidean(shapelet, series)

    def generate_shapelets(self):
        pass