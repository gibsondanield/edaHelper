# -*- coding: utf-8 -*-
"""
The aim of this programme is to automate the initial steps of exploratory data analysis (EDA) in an effort to help the .
The goal is insight rather than intelligence; the user must know the caveats of each of the methods implemented here.

Created on Thu Sep 10 16:52:07 2015

@author: Daniel D. Gibson
"""
#import threading
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
import scatterplot as scplt
from sklearn_pandas import DataFrameMapper, cross_val_score
#statsmodels imports
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.kernel_density import KDEMultivariate

def strip_col_names(df):
    pass

def recreate_df(df, x, cols):
    pass

def drop_singulars(df):
    '''Iterates through dataframe dropping columns with only one value. Returns dataframe'''
    dropped = []
    for col in df.columns:
        if len(set(df[col]))==1:
            df=df.drop(col,axis=1)
            dropped.append(col)
    return df,dropped

def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation using Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


class Unsupervised(object):
    '''Input 'clean' dataset with anytimestamps conderted to pandas datetimeindex.
    Uses fit methods on objects as opposed to fit_transform.'''
    def __init__(self, features_df, y=None, processes=4):
        
        self.features_df = features_df
        self.features_df.columns = map(unicode, self.features_df.columns)
        self.x = pd.DataFrame(features_df)
        self.y = map(bool, y) #why did I do this?
        self.processes = processes
        self.pool = Pool(processes=processes)
        self.log = ['Initialized object']
        
    def normalize(which_subset='train',**kwargs):
        '''Uses StandardScaler'''
        self.log.append('normalize')
        if which_subset=='train':
            self.sc_train=StandardScaler(**kwargs)
            try:
                self.sc_train.fit(self.x_train)
            except:
                print 'Training data not available. Run train_test_split'
        elif which_subset=='test':
            self.sc_test=StandardScaler(**kwargs)
            try:
                self.sc_test.fit(self.x_test)
            except:
                print 'Test data not available. Run train_test_split'
        elif which_subest == 'all':
            self.sc_te=StandardScaler(**kwargs)
            self.sc_train.fit(self.x)
            
            
    def train_test_split(n_xval_folds=5,holdout=0):
        '''Splits data into test and training sets. Holdout is the proportion of data to be kept in the holdout set'''
        self.log.append('test train split')        
        if holdout:#use indices for this
            ss= ShuffleSplit()
            self.holdout_indices = None
#        self.x_train, self.xholdout,self.y_train,self.y_holdout= train_test_split(self.df,self.y,train_size=holdout)
        if n_xval_folds:
            pass
        
    def categorize(self, max_unique_vars=100, make_dummies=False):
        '''distinguish categorical variables from continuous, make dummy variables for categorical'''
        ''' find blank columns'''
        self.log.append('categorize')
        self.varTypes = {}
        for col in self.features_df.columns:
            if type(features_df[col]) in ['datetime64[ns]','<M8[ns]']:
                self.varTypes[col] = 'Time'
            if set(self.features_df[col])>max_unique_vars:
                self.varTypes[col]='Continuous'
            else:
                self.varTypes[col]='Categorical'
                if make_dummies:
                    pass
        
    def reduce_dimensions(self, ndim=3, methods=['tSVD'], classifier=None, plot2d=True, plot3d=False):
        '''To fit or to fit trainsform, that is the question'''
        self.log.append('reduce_dimensions(ndim= %s , methods=%s, classifier=%s, plot2d=%s, plot3d=%s)' %(ndim,methods,classifier,plot2d,plot3d))
        self.ndim = ndim

        clf = self.rf
        if 'tSVD' in methods:
            # plot scree

            # transform
            self.Tsvd = TruncatedSVD(n_components=ndim)
            self.Tsvd.fit(self.X)
            '''think of a clever way to scatterplot'''
#           if plot2d:
#               scplt.binary(self.tData.T[0],self.tData.T[1],None,twoD=plot2d, threeD=plot3d) include transform
            # plot transformed data with labels
            # print component values
        if 'PCA' in methods:
            self.Pca = PCA(n_components=ndim)
            self.Pca.fit(self.X)

    def cluster(self, methods=['kmeans']):
        self.log.append('cluster')
        

    def plot_clusters(self):
        self.log.append('plot clusters')


class BinaryClassification(Unsupervised):

    def __init__(self, features_df, y, processes=4):
        super(BinaryClassification, self).__init__(features_df, y, processes)
        self.rf = RandomForestClassifier(class_weight='auto')
        self.rf.fit(self.X, self.y)
#        self.clustering_methods = [kmeans, kNN]
#        self.dim_reduc_methods = [SVD, PCA]

    def plot_kdes(self, bandwidth=.4, n_features=9, alpha=.10):
        '''Uses various methods (RF feature importance, Two-tailed hypothesis testing) to identify variables of potential interest and plot them using a kde. Bandwith may be changed but defaults to ?. P-values are shown for two-tailed hypothesis test'''
        self.log.append('plot_kdes')        
        # run random forest to get feature importance
        features = self.rf.feature_importances_.argsort()[:n_features]
        self.rf_importances = self.rf.feature_importances_[features]
        plt.figure()
        for i, v in enumerate(self.features_df.columns):
            plt.subplot(n_features / 3 + 1, 3, i)
#             print type(self.X)
            plt.plot(kde_statsmodels_m(self.features_df[self.y][v], np.linspace(
                0, 12, 2000), bandwidth=bandwidth), label='True '+v)
            plt.plot(kde_statsmodels_m(self.features_df[map(lambda x:not x, self.y)][
                     v], np.linspace(0, 12, 2000), bandwidth=bandwidth), label='False '+v)
        plt.legend(loc=0)
        plt.tight_layout()

    def classify(self, grid_density=.02, holdout_set=False, n_folds=5):
        self.log.append('classify')

    def correlated_features(self, n_features, plot=True):
        self.log.append('correlated_features')

    def plot_rocs(self):
        self.log.append('plot_rocs()')
        

    def plot_decision_tree(self, max_splits):
        '''http://scikit-learn.org/stable/modules/tree.html'''
        self.log.append('plot_decision_tree')

    def plot2d_gridsearch_heatmap():
        '''http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html'''
        self.log.append('plot2d_gridsearch_heatmap')

    def plot3D_predictions(self, features_to_use='all', max_plots=3, max_n_features=9, models=['SVC', 'GaussianNB'], plot_decision_boundary=False):
        '''Plots cross section of decision boundary using mean values for missing dimension'''

        '''3-D Scatterplots normalized data in groups three dimensions at a time with True/False Positives/Negatives as colours.
        TP = Green
        TN = Blue
        FP = Yellow
        FN = Orange
        Plots surfaces for Some things?'''
        '''rf_important, '''
        self.log.append('plot3D_predictions')
        self.models = models
        self.svc_kernels = ["linear", "rbf"]
        scplt.binary(self.features_df, self.y, self.rf.predict(self.X))

    def plot2d_predictions(self):
        self.log.append('plot2d_predictions()')

    def compare_model_performance(self):
        self.log.append('compare_model_performance()')

    def max_profit(self, cost_benefit_matrix):
        self.log.append('max_profit')

class Classification(Unsupervised):

    def __init__(self, features_df, y, processes=4):
        super(Classification, self).__init__(features_df, y, processes)
        self.n_classes =  len(set(y))
        
    def print_class_proportions():
        print y.value_counts()
        y.hist()


class Regression(Unsupervised):

    def __init__(self, features_df, y, processes=4):
        super(BinaryClassification, self).__init__(features_df, y, processes)

    def plot_against_y(function=None):
        '''Where colour is squared error or some other var'''
        scplt.scatterRegression()
        
    def linreg():
        '''Uses statsmodels OLS for linear regression. Automatically inserts constant'''
        x=np.hstack((x,np.ones((len(x),1))))
        lm = sm.OLS(endog=ytrain,exog=xtrain,hasconst=1).fit()
        
    def plot_residuals():
        self.log.append('plot_residuals')
    
    def heteroscedacity_check():
        self.log.append('')


class Timeseries:
    

    def plot():
        '''Plots:
        Different Bands of timeseries
        FFT abs
        periodogram
        autocovariance

        '''

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X, y = make_classification(n_features=10, n_samples=1000, n_informative=5,
                               n_clusters_per_class=3, n_redundant=0, hypercube=True, flip_y=.5)
    X = pd.DataFrame(X)
    bc = BinaryClassification(X,y)
#    l = pd.read_csv('../deprivationProject/licwData.csv')
#    l = l.drop(u'Unnamed: 0', axis=1)
#    l = l.drop(0, axis=0)
#    d = pd.read_csv('../deprivationProject/deprivationColumn').True
#    bc = BinaryClassification(l, d)
    bc.reduce_dimensions()
    bc.plot_kdes()
#    bc.plot_predictions()
