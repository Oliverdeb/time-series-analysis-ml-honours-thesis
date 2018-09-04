from scipy.stats import zscore
from numpy import std

class Shapelet:

    def __init__(self, shapelet, index=None, dataset_name=None, color=None, id=1, quality=0):
        self.start_index = index
        self.shapelet = shapelet
        self.color = color
        self.std = std(shapelet)
        self.of_same_class = None
        if self.std == 0:
            print ("STD OF 0 detected for {}, ignoring".format(shapelet, ))        
        else:
            self.std_shapelet = zscore(shapelet)
        self.id = id
        self.quality = quality
        self.dataset_name = dataset_name
    
    def of_same_class_objs(self, shapelet_dict, n=None):
        if not self.of_same_class:
            return []

        if n is not None:
            result = []
            i = 0
            for id in self.of_same_class:
                if i == n:
                    return result
                result.append(shapelet_dict[id])
                i += 1
            return result

        return [shapelet_dict[id] for id in self.of_same_class]

    def __lt__(self, other):
        return self.quality < other.quality

    def __gt__(self, other):
        return self.quality > other.quality

    def to_csv_standardise(self):
        self.shapelet = zscore(self.shapelet)
        return self.to_csv()
    
    def to_csv_offset_0(self):
        offset = self.shapelet[0]
        self.shapelet = self.shapelet - offset
        csv = "0"

        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv

    def to_csv(self):
        csv = str(self.shapelet[0])
        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv
    
    def __sub__(self, other):
        # diff = - (other.shapelet[0] - self.shapelet[0])
        # other_ = other.shapelet + diff if diff != 0 else other.shapelet

        # should this use standardized shape or not?
        shorter = len(self.std_shapelet) if \
            len(self.std_shapelet) < len(other.std_shapelet) \
                else len(other.std_shapelet)
        
        return abs(self.std_shapelet[:shorter] - other.std_shapelet[:shorter])
        # return abs(self.std_shapelet - other.std_shapelet)
            
    def __matmul__(self, other):
        return sum(self - other)

    def __str__(self):
        return "id: %d, shapelet: %s, of_class: %s" % (self.id,  str(self.shapelet), str(self.of_same_class))
