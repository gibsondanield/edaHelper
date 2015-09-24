import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import random as rd

def tfpn(y_true,y_pred, max_points = 100):
    y_false=pd.DataFrame(map(lambda x:not x,y_true))
    y_pred_false =pd.DataFrame( map(lambda x:not x,y_pred))
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)
    tp = map(bool,y_true*y_pred)
    tn =map(bool,(y_false)*(y_pred_false))
    fp =map(bool,y_false*y_pred)
    fn =map(bool,y_true*(y_pred_false))
    return tp, tn, fp, fn

def scatteBinary(data, y_true, y_pred, feature_names=None, threeD=False, max_points = 100, max_plots=3, figure_no=1, decision_fxn=None):
    '''Randomly selects max_points data points to be plotted from each category'''
    '''feature names must be nested list of lists of 3 features'''    
    colors = ['g','b','r','y']    
    data = pd.DataFrame(data)
    tp, tn, fp, fn = tfpn(y_true,y_pred, max_points)
#    print 'scattering1', data.info()
    print 'tp, tn, fp, fn = ', sum(TP), sum(TN), sum(FP), sum(FN)
    for i in range(max_plots):
        
#        print 'scattering2'
        fig = plt.figure(figure_no+i)
        if threeD:
            ax = fig.add_subplot(111, projection='3d')
            cols = (data.columns[3*i],data.columns[3*i+1],data.columns[3*i+2])
    #        print 'cols ', cols
    #        print 'dims ;',data.columns
            for outcome, c, m in [(TP,'g','o'),(TN,'b','o'), (FP,'r','^'), (FN,'y','^')]:
    #            print 'colour ', c, 'sum outcome ', sum(outcome), outcome
                xs = data[outcome][cols[0]]
    #            print 'xs ',xs
                ys = data[outcome][cols[1]]
    #            print len(ys)
                zs = data[outcome][cols[1]]
                ax.scatter(xs, ys, zs, c=c, marker=m)
            
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_zlabel(cols[2])
            
        else: #make 2d plot
            plt.scatter(data.T[0],data.T[1])
        
        if decision_fxn:
            '''plot decision boundary x-section'''
            pass
        
        scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 'o')
        scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['label1', 'label2'], numpoints = 1)
        plt.show()
        
def scatterRegression():
    pass

if __name__ == '__main__':
    pass