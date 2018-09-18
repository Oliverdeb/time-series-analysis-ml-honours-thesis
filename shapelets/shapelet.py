from scipy.stats import zscore
from numpy import std

class Shapelet:

    """
    Shapelet class containing all information about a shapelet
    """

    def __init__(self, shapelet=None, std_shapelet=None, index=None, dataset_name=None, color=None, id=1, quality=0):
        self.start_index = index
        self.shapelet = shapelet
        self.color = color
        self.id = id
        self.quality = quality
        self.dataset_name = dataset_name
        self.std = std(shapelet) if shapelet is not None else 0
        if std_shapelet is not None:
            self.std_shapelet = std_shapelet
            return
        self.of_same_class = None
        if self.std == 0:
            print ("STD OF 0 detected for {}, ignoring".format(shapelet, ))        
        else:
            self.std_shapelet = zscore(shapelet)
        
    
    def of_same_class_objs(self, shapelet_dict, n=None):
        """
        Function that returns all instances of this class, using the shapelet dictionary, as only the IDs of the shapelets are stored
        """
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
        """
        Function that converts the standardized shapelet array into a CSV
        """
        self.shapelet = zscore(self.shapelet)
        return self.to_csv()
    
    def to_csv_offset_0(self):
        """
        Function that converts the shapelet array into a CSV, offseting the array to 0

        Example: if shapelet = [2,1,3]
        0,-1,1 will be returned
        """
        offset = self.shapelet[0]
        self.shapelet = self.shapelet - offset
        csv = "0"

        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv

    def to_csv(self):
        """
        Function that converts the shapelet array into a CSV
        """
        csv = str(self.shapelet[0])
        for elem in self.shapelet[1:]:
            csv += ' ' + str(elem)
        return csv
    
    def __sub__(self, other):
        """
        Subtraction operator between shapelet objects, used in MATMUL operator (see below), to calculate distances between shapelets
        """
        # diff = - (other.shapelet[0] - self.shapelet[0])
        # other_ = other.shapelet + diff if diff != 0 else other.shapelet

        # should this use standardized shape or not?
        shorter = len(self.std_shapelet) if \
            len(self.std_shapelet) < len(other.std_shapelet) \
                else len(other.std_shapelet)
        
        if self.shapelet is None:
            return abs(self.std_shapelet[:shorter] - zscore(other.shapelet[:shorter]))
        if other.shapelet is None:
            return abs(zscore(self.shapelet[:shorter]) - other.std_shapelet[:shorter])
        # restandardize the up to length of shorter shape to not bias         
        return abs(zscore(self.shapelet[:shorter]) - zscore(other.shapelet[:shorter]))
        # return abs(self.std_shapelet - other.std_shapelet)
            
    def __matmul__(self, other):
        """
        Matrix multiplication operator is overriden and used instead to calculate the sum of subtracting two shapelets

        Example:
        
        s1 = [1,2]
        s2 = [2,1]
        s1 @ s2 = sum([1,2] - [2,1]) = sum([1,1])  = 2
        """
        return sum(self - other)


    def sum_dist_entire_class(self, other, shape_dict):
        """
        Function that calculates the sumdist between a shape and every instance of another shape
        """
        sum_dist = other @ self
        for shape in self.of_same_class_objs(shape_dict):
            sum_dist += other @ shape
        return sum_dist

    def __str__(self):
        return "id: %d, shapelet: %s, of_class: %s" % (self.id,  str(self.shapelet), str(self.of_same_class))
