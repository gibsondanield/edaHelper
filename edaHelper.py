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
#from sklearn_pandas import DataFrameMapper, cross_val_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.kernel_density import KDEMultivariate
#from bokeh.charts import Scatter, show
import seaborn as sns
import networkx as nx
import pprint

p=Pool(cpu_count())

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

continuous = ['int','int32','int64','float','float64']

def make_appropriate_plot(
        x_name,
        y_name,
        df,
        z_name=None,
        categorical_var=None,
        continuous_var=None,
        palette='Greens_d',
        context='talk'):
    '''if x_name and y_name are the same, will plot a histogram or KDE'''
    x_dtype, y_dtype = df[x_name].dtype.name, df[y_name].dtype.name
    sns.set_context(context)
    #plt.figure()
    if z_name==None:
        print x_dtype, ' vs. ', y_dtype
        plt.title(x_dtype + ' vs. ' + y_dtype)
        if x_name == y_name:
    #        sns.kdeplot(data=df[y_name])
            if  y_dtype in ['category', 'bool']:
                # plot histogram of count of each category
                try:
                    sns.distplot(df[y_name], kde=False)
                except:
                    print 'error, ',x_name
            elif  y_dtype in ['int64', 'float64']:
                # label
                # sns.distplot(self.df[j],hist=False,label=j)
                try:
                    sns.kdeplot(data=df[y_name])
                except:
                    print 'error, ',x_name
        elif 'datetime' in str(df[x_name].dtype):
            print 'datetime'
            df[[x_name, y_name]].plot()
        elif  y_dtype in ['category', 'bool']:
            if x_dtype in ['category', 'bool']:
                sns.countplot(x=x_name, hue=y_name, data=df,
                              palette=palette)
                #or sns.clustermap
            elif x_dtype in ['int64', 'float64']:
                if  y_dtype == 'category':
                    sns.violinplot(x=x_name, y=y_name, data=df)
                else:
                    sns.violinplot(x=y_name, y=x_name, data=df,split=True,orient="V")
        elif  y_dtype in ['int64', 'float64']:
            if x_dtype in ['category', 'bool']:
                sns.boxplot(x=x_name, y=y_name, data=df)
            elif x_dtype in ['int64', 'float64']:
                # include lin reg and colours/shapes for categories
                sns.jointplot(x=x_name, y=y_name, data=df)#,stat_func=sns.stats.entropy)#lambda x,y:sp.spatial.distance.pdist(zip(x,y), 'correlation'))
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


        plt.title(x_name+ ' vs. '+ y_name+ ' vs. '+ z_name+ ' | '+ x_dtype + ' vs. ' + y_dtype + ' vs. ' + z_dtype)


class Unsupervised(object):
    '''df is PANDAS data frame,
    y is name of target variable (string),
    processes is number of cores for parallelization (int),
    verbose prints intermediate steps in methods,

    Input 'clean' dataset with timestamps converted to pandas datetimeindex.
    '''

    def __init__(self, df, y=None, processes=4, verbose=True):

        self.df = pd.DataFrame(df)
        self.y = y
        self.processes = processes
        self.verbose = verbose
        self.pool = Pool(processes=processes)
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
            palette="Greens_d"):
        '''correlation, mutual information'''
        if cols is None:
            cols = self.vars_of_interest

        for k, i in enumerate(cols[:limit]):
            for j in cols[k:]:
                print 'plotting', k, i, j
                make_appropriate_plot(i, j, self.df, palette)

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
            use_vars_of_interest=True,
            limit=50,
            **appropriate_plot_kwargs):
        '''Creates plots of every variable against the input variable'''
        if use_vars_of_interest:
            for feature in self.vars_of_interest:
                make_appropriate_plot(
                    feature, variable, self.df, **appropriate_plot_kwargs)
        else:
            for feature in self.df.columns[:limit]:
                make_appropriate_plot(
                    feature, variable, self.df, **appropriate_plot_kwargs)

    def dCorr(self, features=None):
        '''scipy correlation function is returning values greater than 1'''
        if features is None:
            features = self.vars_of_interest
        l = len(features)
        arr = np.empty((l, l))
        arr[np.triu_indices(l, 1)] = sp.spatial.distance.pdist(
            self.df[features].values.T, 'correlation')
        self.correlation_distance = pd.DataFrame(
            arr, index=features, columns=features)

    def pursuit_curve(self):
        pass


class Classification(Unsupervised):
    '''Classification object inherits from unsupervised object. Use with binary dependent variable'''

    def __init__(self, x, y, models=[
                 RandomForestClassifier, SVC], processes=4):
        super(Classification, self).__init__(x, y, processes)

        self.rf = RandomForestClassifier(class_weight='auto')
        self.n_classes = self.df[self.y].unique().size
        self.models = models

    def fit(self, **kwargs):
        self.fit_models = []
        for model in self.models:
            self.fitted_models.append(
                model(**kwargs).fit(self.df[self.vars_of_interest], df[[self.y]]))

    def classify(self, grid_density=.02, holdout_set=False, n_folds=5):
        self.log.append('classify')

    def correlated_features(self, n_features, plot=True):
        self.log.append('correlated_features')

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

    def __init__(self, x, y, models=[RandomForestRegressor, SVR], processes=4):
        super(Regression, self).__init__(x, y, processes)
        self.models = models

    def fit(self, **kwargs):
        self.fit_models = []
        for model in self.models:
            self.fitted_models.append(
                model(**kwargs).fit(self.df[self.vars_of_interest], self.df[self.y]))

    def plot_against_y(self, function=None):
        '''Where colour is squared error or some other var'''
        # do linked plots here
        cat, cont, time = cat_cont_time(self.df[self.vars_of_interest])
#        cat = self.df.columns[self.df.dtypes=='category']
#        cont =  self.df.columns[self.df.dtypes=='float64']
        # first continuous
        fig, axs = plt.subplots(nrows=1, ncols=len(cont), sharey=True)
        for ax, col in zip(axs.flat, cont):
            sns.regplot(x=col, y=self.y, data=self.df, ax=ax)
#        g = sns.lmplot(x="total_bill", y=self.y, data=self.df)
        # then categorical

        fig, axs = plt.subplots(nrows=1, ncols=len(cat), sharey=True)
        for ax, col in zip(axs.flat, cat):
            sns.violinplot(x=col, y=self.y, data=self.df, ax=ax)

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
    #make_appropriate_plot('time', 'sine', fake_ts)
    make_appropriate_plot('hp','displ',a.df,z_name='cyl')
    plt.figure()
    make_appropriate_plot('hp','displ',a.df,z_name='accel')

