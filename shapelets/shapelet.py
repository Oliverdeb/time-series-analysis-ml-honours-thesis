from scipy.stats import zscore
from numpy import std

class Shapelet:

    def __init__(self, shapelet, index, dataset_name, color, id=1, quality=0):
        self.start_index = index
        self.shapelet = shapelet
        self.color = color
        self.std = std(shapelet)
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
            
    def __str__(self):
        return "start index" + str(self.start_index) + " " +  str(self.shapelet)
