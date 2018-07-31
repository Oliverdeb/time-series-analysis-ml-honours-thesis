from trend_lines.line import Line
from trend_lines.point import Point

class Utils:

    @staticmethod
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

        
    @staticmethod
    def mse(arr1, arr2):
        sum = 0
        for x in range(0, len(arr1)):
            sum += ((float(arr1[x].price)-float(arr2[x].price))**2)
        sum = sum/(len(arr1)-1)
        return sum
    
    @staticmethod
    def read_file(file_name):
        file = open(file_name, 'r')
        i = 0
        points = []
        for i, line in enumerate(file):
            points.append(Point(i, line.rstrip("\n"), i-9))
        return points
    
    @staticmethod
    def slope_calculator(points):
        sum = 0
        for i, p in enumerate(points[1:], start=1):
            if p!=None:
                sum += float(points[i].price) - float(points[i-1].price)
        return sum/(len(points)-1)

    @staticmethod
    def top_down_segmentation(points, error):
        lines = []
        lines.append(Line(Utils.slope_calculator(points), len(points), 0, points[0].price))
        
        still_error = True
        i = 0
        while still_error:
            still_error = False
            for i, line in enumerate(lines):
                mse = Utils.mse(points[line.start:line.start+line.duration], line.lineArray())
                if mse > error:
                    twoNewLines = Utils.split_line_in_two(line, points)
                    lines[i] = twoNewLines[0]
                    lines.insert(i+1, twoNewLines[1])
                    still_error = True
                    break
        return lines

    @staticmethod
    def bottom_up_segmentation(points, error):
        lines = []
        for i, x in enumerate(points[:-1]):
            lines.append(Line(Utils.slope_calculator(points[i:i+2]), 1, i, x.price))

        still_error = True
        while still_error:
            still_error = False
            join = Utils.find_least_mse_join_lines(points, lines)
            if join[1] < error:
                still_error = True
                lines[join[0]] = Utils.join_two_lines(points, lines[join[0]], lines[join[0]+1])
                del lines[join[0]+1]
        return lines

    @staticmethod
    def join_two_lines(points, line1, line2):
        slope = (float(points[line2.start+line2.duration].price)-float(points[line1.start].price))/2
        line = Line(slope, line2.start+line2.duration-line1.start, line1.start, points[line1.start].price)
        return line

    @staticmethod 
    def find_least_mse_join_lines(points, lines):
        min_index = 0
        min_mse = 1000000000
        for i, line in enumerate(lines[:-1]):
            line2 = lines[i+1]
            joined_lines_points = points[line.start:line.start+line.duration+line2.duration]
            slope = (float(points[line2.start+line2.duration].price)-float(points[line.start].price))
            new_line = Line(slope, line2.start+line2.duration-line.start-1, line.start, points[line.start].price)
            # print("LENGTHS", len(new_line.lineArray()), len(joined_lines_points))
            mse = Utils.mse(new_line.lineArray(), joined_lines_points)
            if(mse<min_mse):
                min_mse = mse
                min_index = i
        return [min_index, min_mse]
            
    @staticmethod
    def split_line_in_two(line, points):
        twoLines = []
        twoLines.append(Line(Utils.slope_calculator(
            points[line.start:line.start+int(line.duration/2)]), 
            int(line.duration/2), 
            line.start, 
            line.c))
        twoLines.append(Line(
            Utils.slope_calculator(points[line.start+int(line.duration/2):line.start+line.duration]), 
            int(line.duration/2),
            line.start+int(line.duration/2), 
            points[line.start+int(line.duration/2)].price))
        return twoLines

    @staticmethod
    def sliding_window_segmentation(points, error):
        lines = []
        lines.append(None)
        start_point = points[0]
        first = True
        for i, current_point in enumerate(points[1:], start=1):
            current_point = points[i]
            if(lines[len(lines)-1]==None):
                if first == True:
                    lines[len(lines)-1] = Line(float(points[i].price)-(float(points[i-1].price))/2, 1, 0, start_point.price)
                    first = False
                else:
                    lines[len(lines)-1] = Line(float(points[i].price)-(float(points[i-1].price))/2, 1, i-1, points[i-1].price)
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
    
    @staticmethod
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
                # print(x)
                p = [(l.start, float(l.c)), (l.start+l.duration, float(x))]
                segments.append(p)
                # print(segments[len(segments)-1])

        
        colors = [(1,0,0,1), (1,1,0,1), (0,0,1,0)]
        c = []
        for x in line:
            c.append(colors[0])
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
    
    @staticmethod
    def print_array(arr):
        for a in arr:
            print(a)