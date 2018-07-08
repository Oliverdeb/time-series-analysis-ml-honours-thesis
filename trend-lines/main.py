class Utils:
    def moving_average(points):
        cumsum, moving_aves = [0], []
        N = 10
        for i, x in enumerate(points, 1):
            cumsum.append(cumsum[i-1] + float(x.price))
            if i>=N:
                x.price = (cumsum[i] - cumsum[i-N])/N
                #can do stuff with moving_ave here
                moving_aves.append(x)
        
        return moving_aves

    def mse(arr1, arr2):
        sum = 0
        for x in range(0, len(arr1)):
            sum += ((float(arr1[x].price)-float(arr2[x].price))**2)
        sum = sum/(len(arr1)-1)
        return sum

    def read_file(file_name):
        file = open(file_name, 'r')
        i = 0
        points = []
        for i, line in enumerate(file):
            points.append(Point(i, line.rstrip("\n"), i-9))
        return points

    def slope_calculator(points):
        sum = 0
        for i, p in enumerate(points[1:], start=1):
            if p!=None:
                sum += float(points[i].price) - float(points[i-1].price)
        return sum/(len(points)-1)


    def sliding_window_segmentation(points, error):
        lines = []
        lines.append(None)
        start_point = points[0]
        first = True
        for i, current_point in enumerate(points[1:], start=1):
            current_point = points[i]
            # print("START POINT", start_point)
            if(lines[len(lines)-1]==None):
                if first == True:
                    lines[len(lines)-1] = Line(float(points[i].price)-(float(points[i-1].price))/2, 1, 0, start_point.price)
                    first = False
                else:
                    lines[len(lines)-1] = Line(float(points[i].price)-(float(points[i-1].price))/2, 1, i-1, start_point.price)
            else:
                newest_line = lines[len(lines)-1]

                check_line = newest_line
                check_line_points = check_line.lineArray()
                check_line_points.append(current_point)
                slope = Utils.slope_calculator(check_line_points)
                check_line = Line(slope, newest_line.duration+1, newest_line.start, newest_line.c)

                mse = Utils.mse(check_line.lineArray(), points[start_point.index:current_point.index+1])
                if(mse<=error or len(check_line_points)<=2):
                    duration = newest_line.duration+1
                    lines[len(lines)-1] = check_line
                else:
                    lines[len(lines)-1].duration = float(newest_line.duration)-1
                    lines.append(None)
                    start_point = current_point
                    i = i-1
        return lines

    def graph(points, line):
        import matplotlib.pyplot as plt
        import numpy as np
        import pylab as pl
        from matplotlib import collections  as mc

        # Print segments 
        print("SEGMENTS")
        segments = []
        for l in line:
            if l != None:
                x = float(l.c)+((l.duration)*l.slope)
                print(x)
                p = [(l.start, float(l.c)), (l.start+l.duration+1, float(x))]
                segments.append(p)
                print(segments[len(segments)-1])

        
        c = []
        for x in lines:
            c.append((1,0,0,1))
        c = np.array(c)

        lc = mc.LineCollection(segments, colors=c, linewidths=2)
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        
        #  Print points
        x = []
        y = []
        for i, point in enumerate(points):
            x.append(float(i))
            y.append(float(point))
            
        plt.scatter(x, y)
        # plt.ylim([0,22])
        plt.show()

    def print_array(arr):
        for a in arr:
            print(a)

class Point:
    time = 0
    price = 0
    index = 0

    def __init__(self, time, price, index):
        self.time = time
        self.price = price
        self.index = index

    def __str__(self):
        return str(self.time) + " : " + str(self.price) + " : " + str(self.index)

class Line:
    slope = 0
    duration = 0
    start = 0
    c = 0
    
    def __init__(self, slope, duration, start, c):
        self.slope = slope
        self.duration = duration
        self.start = start
        self.c = c

    def lineArray(self):
        line = []
        for x in range(self.duration):
            line.append(Point(self.start+x, float(self.c) + float(self.slope*x), self.start+x))
        return line

    def __str__(self):
        return str(self.slope) + " : " + str(self.start) + " : " + str(self.duration+self.start) + " : " + str(self.c)

if __name__ == '__main__':  
    print('\n===== POINTS =====')
    points = Utils.read_file('snp2.csv')
    print(len(points))

    points = Utils.moving_average(points)
    print(len(points))
    Utils.print_array(points)

    print('\n===== LINES =====')
    lines = Utils.sliding_window_segmentation(points, 120)
    Utils.print_array(lines)
    print('\n')

    p = []
    for x in points:
        p.append(x.price)

    Utils.graph(p, lines)