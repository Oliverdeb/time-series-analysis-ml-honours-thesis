from numpy import arange, array, ones
from numpy.linalg import lstsq
import time
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
def draw_plot(data,plot_title):
    plot(range(len(data)),data,alpha=0.8,color='red')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))

def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)

def leastsquareslinefit(sequence,seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    x = arange(seq_range[0],seq_range[1]+1)
    y = array(sequence[seq_range[0]:seq_range[1]+1])
    A = ones((len(x),2),float)
    A[:,0] = x
    (p,residuals, rank, s) = lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p,error)

def stats(name, mse, start_time, segments, points):
    print("\n\n",'-'*len(name),sep='')
    print(name)
    print('-'*len(name),sep='')
    print("MSE\t\t\t: %.2f" % mse)
    print("Segments\t\t: %d" % len(segments))
    print("Run time\t\t: %.2f" % (time.time()-start_time))
    print("MSE (actual)\t\t: %.2f" % mse_calculator(segments, points))
    print('-'*len(name),sep='')

def mse_calculator(segments, points):
    mse = 0
    for segment in segments:
        mse = mse +  leastsquareslinefit(points, (int(segment[0]), int(segment[2])))[1]
    return mse/len(segments)

def convert_to_slope_duration(segments):
    s_d = []
    for s in segments:
        duration = s[2]-s[0]
        slope = (s[3]-s[1])/duration
        s_d.append([slope, duration])
    return s_d