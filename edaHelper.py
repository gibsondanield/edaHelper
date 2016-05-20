# -*- coding: utf-8 -*-
"""
The aim of this programme is to automate some common tasks in exploratory data analysis (EDA) in an effort to save time for the user.
The goal is insight rather than intelligence; the user must know the caveats of each of the methods implemented here.

Created on Thu Sep 10 16:52:07 2015

@author: Daniel D. Gibson
"""
#import threading
from multiprocessing import Pool, cpu_count
import sys
from scipy.stats import chisquare
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.magics import logging
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import normalized_mutual_info_score
#from sklearn_pandas import DataFrameMapper, cross_val_score
import statsmodels.api as sm
#from bokeh.charts import Scatter, show
import seaborn as sns
import networkx as nx
import pprint


def square_matrix_plot(matrix,vmax=1,vmin=0):
    sns.set(style="white")
    corr = 1-matrix

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set_context('talk')
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,  cmap='YlGnBu', vmax=vmax,vmin=vmin,
                square=True,
                linewidths=.5, cbar_kws={"shrink": .9}, ax=ax)
    plt.title('pairwise')
    return f, ax

def strip_col_names(df, suffix='',separator='_'):
    '''makes all columns available as attributes. checks for redundancy and appends append variable to redunant names'''
    df.columns = p.map(lambda s:s.replace(' ',separator),df.columns)
    return df


def recreate_df(df, x, columns):
    for i, col in enumerate(columns):
        df[col] = x.T[i]
    return df


def add_const(numpy_ndarray, const=1.0):
    return np.hstack((numpy_ndarray, np.ones((len(numpy_ndarray), const))))


def drop_singulars(df):
    '''Iterates through dataframe dropping columns with only one value. Returns dataframe'''
    dropped = []
    for col in df.columns:
        if df[col].unique().size == 1:
            df = df.drop(col, axis=1)
            dropped.append(col)
    return df, dropped

def cat_cont_time(df):
    '''returns lists of which variables are categorical, continuous or temporal in df
    O(n) runtime where n is number of columns in df'''
    cat, cont, time = [], [], []
#    print '.....................'
    for col in df.columns:
        #        print col
        if df[col].dtype == float or df[col].dtype == int:
            cont.append(col)

        elif str(df[col].dtype) in ['category', 'bool']:
            cat.append(col)
#            print col
        # https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
        elif str(df[col].dtype) in ['datetime64[ns]', '<M8[ns]']:
            time.append(col)

    return cat, cont, time

continuous = {int,float}
categorical = {'category', np.dtype(bool)}
temporal = {'datetime64[ns]', np.dtype('<M8[ns]')}

def make_appropriate_plot(
        x_name,
        y_name,
        df,
        z_name=None,
        title=None,
        ax=None,
        palette='Greens_d',
        context='talk'):
    '''if x_name and y_name are the same, will plot a histogram or KDE
    returns matplotlib axes'''
    x_dtype, y_dtype = df[x_name].dtype.name, df[y_name].dtype.name
    if ax==None:
        #plt.figure()
        ax=plt.subplot()
    sns.set_palette(palette)
    sns.set_context(context)
    if z_name==None:
        print x_dtype, ' vs. ', y_dtype
        if title:
            plt.title(title)
        else:
            plt.title(x_name + ' vs. ' + y_name)
        if x_name == y_name:
    #        sns.kdeplot(data=df[y_name])
            if  y_dtype in ['category', 'bool']:
                # plot histogram of count of each category
                try:
                    sns.distplot(df[y_name], kde=False,ax=ax)
                except:
                    print 'error, ',x_name
                    print sys.exc_info()
            elif  y_dtype in ['int64', 'float64']:
                # label
                # sns.distplot(self.df[j],hist=False,label=j)
                try:
                    sns.kdeplot(data=df[y_name],ax=ax,vertical=True)
                except:
                    print 'error, ',x_name
                    print sys.exc_info()
        elif 'datetime' in str(df[x_name].dtype):
            print 'datetime'
            df[[x_name, y_name]].plot()
        elif  y_dtype in ['category', 'bool']:
            if x_dtype in ['category', 'bool']:
                sns.countplot(x=x_name, hue=y_name, data=df,
                              palette=palette,ax=ax)
                #or sns.clustermap
            elif x_dtype in ['int64', 'float64']:
                if  y_dtype == 'category':
                    sns.violinplot(x=x_name, y=y_name, data=df,ax=ax)
                else:
                    sns.violinplot(x=y_name, y=x_name, data=df,split=True,orient="V",ax=ax)
        elif  y_dtype in ['int64', 'float64']:
            if x_dtype in ['category', 'bool']:
                sns.boxplot(x=x_name, y=y_name, data=df,ax=ax)
            elif x_dtype in ['int64', 'float64']:
                # include lin reg and colours/shapes for categories
                try:
                    sns.regplot(x=x_name, y=y_name, data=df,ax=ax,fit_reg=False)#,stat_func=sns.stats.entropy)#lambda x,y:sp.spatial.distance.pdist(zip(x,y), 'correlation'))
                except:
                    print 'jointplot error '
                    print sys.exc_info()
    else:
        z_dtype= df[z_name].dtype.name
        print x_dtype, ' vs. ', y_dtype, ' vs. ', z_dtype

        '''
        let's start out assuming there is a continuous var for x. if not continuous, switch with y
        '''
        if x_dtype in ['int64', 'float64']:
            if y_dtype in ['int64', 'float64']:
                if z_dtype in ['category', 'bool']:
                    sns.lmplot(x_name, y_name, data=df, hue=z_name)
                if z_dtype in ['int64', 'float64']:
                    plt.scatter(df[x_name],df[y_name],c=df[z_name],cmap='Greens')

            if y_dtype in ['category', 'bool']:
                pass
        if title:
            plt.title(title)
        else:
            plt.title(x_name+ ' vs. '+ y_name+ ' vs. '+ z_name+ ' | '+ x_dtype + ' vs. ' + y_dtype + ' vs. ' + z_dtype)
    return ax

class Unsupervised(object):
    '''df is PANDAS data frame,
    y is name of target variable (string),
    processes is number of cores for parallelization (int) (defaults to number of cores on machine),
    verbose prints intermediate steps in methods,

    Input 'clean' dataset with timestamps converted to pandas datetimeindex.
    '''

    def __init__(self, df, y=None, processes=None, verbose=True):

        self.df = pd.DataFrame(df)
        self.y = y
        self.processes = processes if processes else cpu_count()
        self.verbose = verbose
        #self.pool = Pool(processes=processes)
        self.log = ['Initialized object']  # look into logging module
        self.vars_of_interest = self.df.columns[self.df.columns != self.y]


    def set_vars_of_interest(self, columns=None):
        '''sets vars_of_intetest to specified values. Defaults to non-objects.
        columns is list-like'''
        if columns is None:
            self.df.dtypes.index[self.df.dtypes != 'object']
        else:
            self.vars_of_interest = pd.Index(columns)
        if self.verbose:
            print 'vars_of_interest set to: ', self.vars_of_interest

    # integrate this with everything else using pipeline?%matp
    def normalize(self, which_subset='train', **kwargs):
        '''Uses StandardScaler'''
        self.log.append('normalize')
        if which_subset == 'train':
            self.sc_train = StandardScaler(**kwargs)
            try:
                self.sc_train.fit(self.df_train)
            except:
                print 'Training data not available. Run train_test_split'
        elif which_subset == 'test':
            self.sc_test = StandardScaler(**kwargs)
            try:
                self.sc_test.fit(self.df_test)
            except:
                print 'Test data not available. Run train_test_split'
        elif which_subset == 'all':
            self.sc_all = StandardScaler(**kwargs)
            self.sc_all.fit(self.df)

    def make_dummy_variables(
            self,
            drop_original=True,
            delimiter='_',
            dummy_na=False,
            **kwargs):
        # find some way of dealing with NaN
        print self.vars_of_interest
        categorical_vars = self.df.dtypes.index[self.df.dtypes == 'category']
        for variable in set(categorical_vars).intersection(
                set(self.vars_of_interest)):
            if self.df[variable].dtype == 'category':
                if self.verbose:
                    print 'making dummy variables for: ', variable
                # First we create the columns with dummy variables.
                # Note that the argument 'prefix' means the column names will be
                # prefix_value for each unique value in the original column, so
                # we set the prefix to be the name of the original variable.
                dummy_df = pd.get_dummies(
                    self.df[variable],
                    prefix=variable,
                    dummy_na=dummy_na,
                    **kwargs)
                if self.verbose:
                    print variable, ' has value ', dummy_df.columns[0], ' when ', dummy_df.columns[1:].values, 'equal zero'
                dummy_df = dummy_df.drop(dummy_df.columns[0], axis=1)

                # Remove old variable from dictionary.
                if drop_original:
                    self.df.pop(variable)
                    self.vars_of_interest = self.vars_of_interest.drop(
                        variable)

                self.df = self.df.join(dummy_df)
                self.vars_of_interest = self.vars_of_interest.append(
                    dummy_df.columns)
        # self.categorize(max_unique_vars=2)

    def convert_for_statsmodels(self):
        '''Convert to float'''
        pass

    def train_test_split(self, n_xval_folds=5, holdout=0):
        '''Splits data into test and training sets. Holdout is the proportion of data to be kept in the holdout set'''
        self.log.append('test train split')
        self.n_xval_folds = n_xval_folds
        self.holdout = holdout
        if holdout:  # use indices for this

            ss = ShuffleSplit()
            self.holdout_indices = None
#        self.df_train, self.dfholdout,self.y_train,self.y_holdout= train_test_split(self.df,self.df[y],train_size=holdout)
        if n_xval_folds:
            pass

    def categorize(self, max_unique_vars=10):
        '''distinguish categorical variables from continuous. only int and float64 are considered continuous variables'''
        self.log.append('categorize')
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'float', 'int']:
                n = self.df[col].unique().size
                if self.verbose:
                    print col, 'has ', n, ' unique values'
                if n == 2:  # convert to bool
                    self.df[col] = self.df[col].astype(bool)
                elif n <= max_unique_vars:
                    self.df[col] = self.df[col].astype('category')
        if self.verbose:
            print self.df.info()

    def plot_all(
            self,
            cols=None,
            prioritization_method='correlation',
            limit=10,
            **appropriate_plot_kwargs):
        '''
        if cols=None, plots vars_of_interest
        prioritizations: correlation, mutual information'''

        if cols is None:
            cols = self.vars_of_interest

        for k, i in enumerate(cols[:limit]):
            plt.figure(k)
            for j in cols[k:]:
                print 'plotting', k, i, j
                ax=plt.subplot()
                make_appropriate_plot(i, j, self.df, ax=ax,**appropriate_plot_kwargs)

    def tfidf_convert(self, column_name, **kwargs):
        '''Converts all text data to tfidf vectors, updates vars_of_interest with new columns'''
        self.log.append('tfidf')

    # fix this as per other modeling methods
    def reduce_dimensions(
            self,
            ndim=3,
            objects=['tSVD'],
            classifier=None,
            plot2d=True,
            plot3d=False,
            whiten=True):
        '''uses fit method on whichever objects are specified'''
        self.log.append(
            'reduce_dimensions(ndim= %s , objects=%s, classifier=%s, plot2d=%s, plot3d=%s)' %
            (ndim, methods, classifier, plot2d, plot3d))
        self.ndim = ndim

        clf = self.rf
        if 'tSVD' in objects:
            # plot scree

            # transform
            self.Tsvd = TruncatedSVD(n_components=ndim)
            self.Tsvd.fit(self.df)
            '''think of a clever way to scatterplot'''
#           if plot2d:
#               scplt.binary(self.tData.T[0],self.tData.T[1],None,twoD=plot2d, threeD=plot3d) include transform
            # plot transformed data with labels
            # print component values
        if 'PCA' in objects:
            self.Pca = PCA(n_components=ndim, whiten=whiten)
            self.Pca.fit(self.df)

    def cluster(self, objects=['kmeans']):
        self.log.append('cluster')

    def cov_cat(self):
        '''See http://arxiv.org/pdf/0711.4452.pdf
        https://en.wikipedia.org/wiki/List_of_analyses_of_categorical_data'''
        self.log.append('cov_cat()')

    def plot_clusters(self):
        self.log.append('plot clusters')

    def only(self):
        '''picks unique values from categorigal variables that only have unique values'''
        cat, cont, time = cat_cont_time(self.df)  # [self.vars_of_interest]
        if cat:
            self.only_list = []
            for col in cat:
                for value in self.df[col].unique():
                    dummydf = self.df[self.df[col] == value].drop(col, axis=1)
                    for col2 in dummydf.columns:
                        val2 = dummydf[col2].unique()
                        if val2.size == 1:
                            self.only_list.append(
                                '%s in column %s only has value %s in column %s' %
                                (value, col, val2, col2))
            for i in self.only_list:
                print i

    def corr_graph(self):
        '''Find different graph database'''
        self.Corr_graph = nx.Graph()
        for i, v in enumerate(self.df.corr().columns):
            self.Corr_graph.add_node(v)
            for j, w in enumerate(self.df.corr().columns[i + 1:]):
                print j, w, self.df.corr().values[i][j + 1]
                self.Corr_graph.add_edge(
                    v, w, weight=self.df.corr().values[i][j + 1])
        nx.draw(self.Corr_graph)

    def _return_categorical_and_boolean_columns(self):
        cols = self.df.dtypes.index[self.df.dtypes == bool].append(
            self.df.dtypes.index[self.df.dtypes == 'category'])
        try:
            cols.drop(self.y)
        except:
            pass
        if self.verbose:
            print '_return_categorical_and_boolean_columns:'
            print cols
        return cols

    def significance_test(self, target_var=None, p=.05,
                          multivariate_correction=None):
        ''' Runs chi2 on categorical and boolean variables'''
        self.log.append('')

        if target_var is None:
            target_var = self.y

        if self.df[target_var].dtype == bool:
            target_proportion = sum(
                self.df[target_var] == 1) / float(self.df[target_var].size)
            self.chi_2_results = {}
            columns = self._return_categorical_and_boolean_columns()
            for col in columns:
                column_fraction = self.df[self.df[target_var]][col].value_counts(
                ).values / p.map(float, self.df[col].value_counts().values)
                chi_2 = chisquare(column_fraction, [
                                  target_proportion for _ in column_fraction])

                if self.verbose:
                    print col, 'column fraction = ', column_fraction
                    print 'chi_2 = ', chi_2
                    pprint.pprint(chi_2)
                self.chi_2_results[col] = chi_2
        return self.chi_2_results.values()

    def plot_against_(
            self,
            variable,
            features=None,
            dependent=True,

            limit=10,
            **appropriate_plot_kwargs):
        '''Creates plots of every variable against the input variable'''
        #plt.figure()

        cols= self.return_features(features)[:limit]
        n_plots= len(cols)
        if dependent:
            fig, axs = plt.subplots(nrows=1, ncols=n_plots, sharey=True)
        else:
            fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)
        for ax, col in zip(axs.flat, cols):
            make_appropriate_plot(
                col, variable, self.df,ax=ax, **appropriate_plot_kwargs)
        plt.title(variable + 'vs. all')
        return plt

    def prioritize(features,limit=None,priority=None,high=True,target=None):
        pass

    def return_features(features,limit=None,priority=None,high=True,target=None):
        '''
        high/low
        target var
        mutual info, correlation
        '''
        if priority:
            if priority in self.column_distance[priority]:
                scores=self.column_distance[priority]
            else:
                scores=self.make_distance_matrix(metric=priority)
            #sort scores, return corresponding columns

        if features == None:
            features = self.vars_of_interest
        elif features == 'all':
            features=self.df.columns
        elif features == 'categorical':
            pass
        elif features == 'continuous':
            pass
        elif features == 'temporal':
            pass
        elif features == 'text':
            pass
        return features


    def make_distance_matrix(self, features=None,transpose=True,metric='correlation'):
        '''uses scipy.spatial.distance.pdist to compute given distance metric
        see http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
        stores result as Dataframe in dictionary self.distance[metric]
        transpose=True means that the distance metric is computed between columns'''
        features= self.return_features(features)

        if transpose:
            values=self.df[features].values.T
        else:
            values=self.df[features].values
        l = len(features)
        arr = np.empty((l, l))
        arr[np.triu_indices(l, 1)] = sp.spatial.distance.pdist(
            values, metric=metric)
        if transpose:
            self.column_distance[metric] = pd.DataFrame(
            arr, index=features, columns=features)
            print self.distance[metric]
        else:
            self.row_distance[metric] = pd.DataFrame(
            arr)


    def make_NMI_matrix(self,features=None):
        '''computes normalized mutual information
        stores in self.normalized_mutual_information'''
        features= return_features(features)
        #make dummy vars without dropping?
        l = len(features)
        arr = np.empty((l, l))
        arr.fill(np.nan)
        for i,series in enumerate(self.df[features].iteritems()):
            for j,series2 in enumerate(self.df[features[i:]].iteritems()):
                #print i,j+i,series[0],series2[0]
                try:
                    arr[i,j+i] = normalized_mutual_info_score(series[1],series2[1])
                    #print arr[j,i]
                except:
                    #print 'except' + str(series) + str(series2)
                    arr[j,i]=-1
                    #print sys.exc_info()
        self.normalized_mutual_information = pd.DataFrame(
            arr.T, index=features, columns=features)

    def pursuit_curve(self):
        pass


class Classification(Unsupervised):
    '''Classification object inherits from unsupervised object. Use with binary dependent variable
    Models are initialized objects'''

    def __init__(self, x, y):
        super(Classification, self).__init__(x, y,cost_benefit_matrix=None,
            models=None,fit_kwargs=None)

#        self.rf = RandomForestClassifier(class_weight='auto',n_jobs=self.processes)
        self.n_classes = self.df[self.y].unique().size
        self.cost_benefit_matrix=cost_benefit_matrix
        self.models=models
        self.fit_kwargs=fit_kwargs

    def fit(self, data_indices=None):
        '''Uses .fit() method on each model
        operates on models in parallel'''
        p=Pool(self.processes)
        p.map_async(lambda x,kwargs: x.fit(self.df[self.vars_of_interest],
                                           df[[self.y]],**kwargs),
                                            zip(self.models,self.fit_kwarg_dicts))
        p.get()
        p.close()

    def classify(self, grid_density=.02, holdout_set=False, n_folds=5):
        self.log.append('classify')


#    def correlated_features(self, n_features, plot=True):
#        self.log.append('correlated_features')

    def plot_rocs(self):
        self.log.append('plot_rocs()')
        for model in self.fitted_models:
            # model.predict_proba
            pass

    def plot_decision_tree(self, **kwargs):
        '''http://scikit-learn.org/stable/modules/tree.html'''
        self.log.append('plot_decision_tree')
        self.dtree = DecisionTreeClassifier(**kwargs)
        self.dtree.fit(self.df[self.vars_of_interest], self.df[self.y])

    def plot2d_gridsearch_heatmap():
        '''http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html'''
        self.log.append('plot2d_gridsearch_heatmap')

    def plot3D_predictions(
            self,
            features_to_use='all',
            max_plots=3,
            max_n_features=9,
            models=[
                'SVC',
                'GaussianNB'],
            plot_decision_boundary=False):
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

    def plot2d_predictions(
            self,
            limit=20,
            plot_decision_boundary=True,
            decision_boundary_method='mean'):
        self.log.append('plot2d_predictions()')
        #plot continuous variables

        #plot categorical and boolean

        #

    def compare_model_performance(self):
        self.log.append('compare_model_performance()')

    def max_profit(self, cost_benefit_matrix):
        self.log.append('max_profit')

    def hist_class_proportions():
        print y.value_counts()
        plt.hist(y)


class Regression(Unsupervised):

    def __init__(self, x, y, models=[RandomForestRegressor, SVR]):
        super(Regression, self).__init__(x, y)
        self.models = models

    def fit(self, **kwargs):
        self.fit_models = []
        for model in self.models:
            self.fitted_models.append(
                model(**kwargs).fit(self.df[self.vars_of_interest], self.df[self.y]))

    def plot_against_y(self, function=None,y_margin=.1,lim=10,context='talk'):
        '''Where colour is squared error or some other var'''
        # do linked plots here
        cat, cont, time = cat_cont_time(self.df[self.vars_of_interest])
#        cat = self.df.columns[self.df.dtypes=='category']
#        cont =  self.df.columns[self.df.dtypes=='float64']
        # first continuous
        cols=cat+cont+time
        cols=cols[:10]
        sns.set_context(context)
        fig, axs = plt.subplots(nrows=1, ncols=len(cols), sharey=True)
        for ax, col in zip(axs.flat, cols):
            if col in cont:
                sns.regplot(x=col, y=self.y, data=self.df, ax=ax)
#        g = sns.lmplot(x="total_bill", y=self.y, data=self.df)
        # then categorical

        #fig, axs = plt.subplots(nrows=1, ncols=len(cat), sharey=True)
        #for ax, col in zip(axs.flat, cat):
            elif col in cat:
                sns.violinplot(x=col, y=self.y, data=self.df, ax=ax)
            else:
                #plot timeseries
                self.df([self.y,col]).plot()
        y_min,y_max=self.df[self.y].min(),(self.df[self.y].max())
        y_range=y_max-y_min
        plt.ylim(y_min-y_margin*y_range,y_max+y_margin*y_range)
#        g = sns.FacetGrid(self.df,col=self.df.columns[self.df.dtypes=='category'],row=self.y,sharey=True)
#        g.map(sns.violinplot)
        return fig

    def linreg(self):
        '''Uses statsmodels OLS for linear regression. Automatically inserts constant'''
        lm = sm.OLS(endog=ytrain, exog=add_const(
            self.df[self.vars_of_interest]), hasconst=1).fit()
        print lm.summary2()

    def plot_residuals(self):
        self.log.append('plot_residuals')
        # sns.residplot
#        https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.residplot.html

LM = logging.LoggingMagics()
# LM.logstart()

if __name__ == '__main__':
    #    from sklearn.datasets import make_classification
    #
    #    x, y = make_classification(n_features=10, n_samples=1000, n_informative=5,
    #                               n_clusters_per_class=3, n_redundant=0, hypercube=True, flip_y=.5)
    #    x = pd.DataFrame(x)
    #    bc = Classification(x,y)

    from bokeh.sampledata.autompg import autompg
    a = Regression(autompg, 'mpg')
    a.categorize()
    a.vars_of_interest = a.vars_of_interest.drop('name')
    a.only()

    titanic = sns.load_dataset("titanic")
    t = Classification(titanic, 'survived')
    t.categorize()
    #t.make_dummy_variables()

    iris = sns.load_dataset("iris")
    i = Unsupervised(iris)
    i.categorize()
    i.only()
#    sns.pairplot(iris)
    gammas = sns.load_dataset("gammas")
    x = np.linspace(0, 15, 310)
    data = np.sin(x)
    fake_ts = pd.DataFrame(data, columns=['sine'])
    fake_ts['time'] = pd.DatetimeIndex(x)
#    make_appropriate_plot('time', 'sine', fake_ts)
#    make_appropriate_plot('hp','displ',a.df,z_name='cyl')
#    plt.figure()
    #ax=make_appropriate_plot('hp','displ',a.df,z_name='accel')

