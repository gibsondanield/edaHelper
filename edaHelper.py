# -*- coding: utf-8 -*-
"""
The aim of this programme is to automate some common tasks in exploratory data analysis (EDA) in an effort to help the .
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
from bokeh.charts import Scatter, show
import seaborn as sns

def strip_col_names(df,append=''):
    '''makes all columns available as attributes. checks for redundancy and appends append variable to redunant names'''
    pass

def recreate_df(df, x, columns):
    for i,col in enumerate(columns):
        df[col]=x.T[i]
    return df

def add_const(numpy_ndarray,const=1):
    return np.hstack((numpy_ndarray,np.ones((len(numpy_ndarray),const))))

def drop_singulars(df):
    '''Iterates through dataframe dropping columns with only one value. Returns dataframe'''
    dropped = []
    for col in df.columns:
        if len(set(df[col]))==1:
            df=df.drop(col,axis=1)
            dropped.append(col)
    return df,dropped

def cat_cont_time(df):
    '''returns lists of which variables are categorical, continuous or temporal in df
    O(n) runtime where n is number of columns in df'''
    cat, cont, time = [],[],[]
    for col in df.columns:
        if df[col].dtype==float or df[col].dtype==int:
            cont.append(col)
            return
        try:
            if df[col].dtype==['category','bool']:
                cat.append(col)
        except:
            pass
        if df[col].dtype in ['datetime64[ns]','<M8[ns]']: #https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
            time.append(col)
            
    return cat, cont, time

class Unsupervised(object):
    '''Input 'clean' dataset with anytimestamps conderted to pandas datetimeindex.
    Uses fit methods on objects as opposed to fit_transform.'''
    def __init__(self, df, y=None, processes=4):

        self.df = pd.DataFrame(df)
        self.y = y 
        self.processes = processes
        self.pool = Pool(processes=processes)
        self.log = ['Initialized object']
        self.vars_of_interest= self.df.columns
        
    def normalize(which_subset='train',**kwargs): #integrate this with everything else using pipeline?%matp
        '''Uses StandardScaler'''
        self.log.append('normalize')
        if which_subset=='train':
            self.sc_train=StandardScaler(**kwargs)
            try:
                self.sc_train.fit(self.df_train)
            except:
                print 'Training data not available. Run train_test_split'
        elif which_subset=='test':
            self.sc_test=StandardScaler(**kwargs)
            try:
                self.sc_test.fit(self.df_test)
            except:
                print 'Test data not available. Run train_test_split'
        elif which_subest == 'all':
            self.sc_all=StandardScaler(**kwargs)
            self.sc_all.fit(self.df)
            
            
    def make_dummy_variables(drop_original=True,**kwargs):
        #Loop over nominal variables.
        for variable in filter(lambda q: self.varTypes[q]=='categorical',
                               self.varTypes.keys()):
     
            #First we create the columns with dummy variables.
            #Note that the argument 'prefix' means the column names will be
            #prefix_value for each unique value in the original column, so
            #we set the prefix to be the name of the original variable.
            dummy_df=pd.get_dummies(self.df[variable], prefix=variable,**kwargs)
     
            #Remove old variable from dictionary.
            if drop_original:
                self.varTypes.pop(variable)
     
            #Add new dummy variables to dictionary.
            for dummy_variable in dummy_df.columns:
                self.varTypes[dummy_variable] = 'Binary'
     
            #Add dummy variables to main df.
            self.df=self.df.drop(variable, axis=1)
            self.df=self.df.join(dummy_df)
    
            
    def train_test_split(n_xval_folds=5,holdout=0):
        '''Splits data into test and training sets. Holdout is the proportion of data to be kept in the holdout set'''
        self.log.append('test train split')        
        self.n_xval_folds= n_xval_folds
        self.holdout = holdout
        if holdout:#use indices for this
        
            ss= ShuffleSplit()
            self.holdout_indices = None
#        self.df_train, self.dfholdout,self.y_train,self.y_holdout= train_test_split(self.df,self.df[y],train_size=holdout)
        if n_xval_folds:
            pass
        
    def categorize(self, max_unique_vars=10):
        '''distinguish categorical variables from continuous. only int and float64 are considered continuous variables'''

        self.log.append('categorize')
#        self.varTypes = {}
        for col in self.df.columns:
#            print self.df[col].dtype
            if self.df[col].dtype in ['object','float64','int']:
                n=len(set(self.df[col]))
                print col,'has ',n,' unique values'
                if n<=max_unique_vars:
                    self.df[col]=self.df[col].astype('category')
#                if n==2: #convert to bool
#                    self.df[col]=self.df[col].astype(bool)

        print self.df.info()
    def plot_all(self,cols=None):
        if cols==None:
            cols=self.df.columns
        for k,i in enumerate(cols):
            for j in cols[k:]:
                if i==j:
                    if str(self.df[j].dtype) in ['category','bool']:
                        pass
                elif str(self.df[j].dtype) in ['category','bool']:
                    if str(self.df[i].dtype) in ['category','bool']:
                        plt.figure()
                        sns.countplot(x=i,hue=j,data=self.df,palette="Greens_d")
                    elif str(self.df[i].dtype) in ['int','float64']:
                        plt.figure()
                        sns.violinplot(x=i,y=j,data=self.df)
                elif str(self.df[j].dtype) in ['int','float64']:
                    if str(self.df[i].dtype) in ['category','bool']:
                        plt.figure()
                        sns.boxplot(x=i,y=j,data=self.df)
                    elif str(self.df[i].dtype) in ['int','float64']:
                        plt.figure()
                        sns.jointplot(x=i,y=j,data=self.df) #fit lin reg???
        
    def column_to_vec(self,**kwargs):
        '''Converts all text data to tfidf vectors'''
        self.log.append('column_to_vec')
        
        
    def reduce_dimensions(self, ndim=3, objects=['tSVD'], classifier=None, plot2d=True, plot3d=False, whiten=True): #fix this as per other modeling methods
        '''uses fit method on whichever objects are specified'''
        self.log.append('reduce_dimensions(ndim= %s , objects=%s, classifier=%s, plot2d=%s, plot3d=%s)' %(ndim,methods,classifier,plot2d,plot3d))
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
        if 'PCA' in methods:
            self.Pca = PCA(n_components=ndim,whiten=whiten)
            self.Pca.fit(self.df)

    def cluster(self, objects=['kmeans']):
        self.log.append('cluster')
        

    def plot_clusters(self):
        self.log.append('plot clusters')
        
    def only(self):
        
        '''picks unique values from categorigal variables that only have unique values'''
        cat, cont, time = cat_cont_time(self.df)#[self.vars_of_interest]
        if cat:
            self.only=[]
            for col in cat:
                for value in set(self.df[col].unique()):
                    dummydf=self.df[self.df[col]==value].drop(col,axis=1)
                    for col2 in dummydf.columns:
                        val2=set(dummydf[col2])
                        if len(val2)==1:
                            self.only.append('%s in column %s only has value %s in column %s' % (value, col,val2,col2)) 
            print self.only


class Classification(Unsupervised):

    def __init__(self, x, y, models=[RandomForestClassifier,SVC], processes=4):
        super(Classification, self).__init__(x, y, processes)
        self.rf = RandomForestClassifier(class_weight='auto')
        self.rf.fit(self.df, self.df[y])
        self.n_classes =  len(set(y))
        self.models = models
#        self.clustering_methods = [kmeans, kNN]
#        self.dim_reduc_methods = [SVD, PCA]
	def fit(self,**kwargs):
		self.fit_models = []
		for model in self.models:
			self.fitted_models.append(model(**kwargs).fit(self.df[self.vars_of_interest],df[self.y]))


    def plot_kdes(self, bandwidth=.4, n_features=9, alpha=.10):
        '''Uses various methods (RF feature importance, Two-tailed hypothesis testing) to identify variables of potential interest and plot them using a kde. Bandwith may be changed but defaults to ?. P-values are shown for two-tailed hypothesis test'''
        self.log.append('plot_kdes')        
        # run random forest to get feature importance
        features = self.rf.feature_importances_.argsort()[:n_features]
        self.rf_importances = zip(self.df.columns,self.rf.feature_importances_[features])
        plt.figure()
        for i, v in enumerate(self.df.columns):
            plt.subplot(n_features / 3 + 1, 3, i)
#             print type(self.df)
            plt.plot(kde_statsmodels_m(self.df[self.df[y]][v], np.linspace(
                0, 12, 2000), bandwidth=bandwidth), label='True '+v)
            plt.plot(kde_statsmodels_m(self.df[map(lambda x:not x, self.df[y])][
                     v], np.linspace(0, 12, 2000), bandwidth=bandwidth), label='False '+v)
        plt.legend(loc=0)
        plt.tight_layout()

    def classify(self, grid_density=.02, holdout_set=False, n_folds=5):
        self.log.append('classify')

    def correlated_features(self, n_features, plot=True):
        self.log.append('correlated_features')

    def plot_rocs(self):
        self.log.append('plot_rocs()')
        for model in self.fitted_models:
        	#model.predict_proba
            pass

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
        scplt.binary(self.df, self.df[y], self.rf.predict(self.df))

    def plot2d_predictions(self):
        self.log.append('plot2d_predictions()')

    def compare_model_performance(self):
        self.log.append('compare_model_performance()')

    def max_profit(self, cost_benefit_matrix):
        self.log.append('max_profit')

    def print_class_proportions():
        print y.value_counts()
        plt.hist(y)

class Regression(Unsupervised):

    def __init__(self, x, y, models=[RandomForestRegressor,SVR],processes=4):
        super(Regression, self).__init__(x, y, processes)
        self.models = models
        

    def fit(self,**kwargs):
		self.fit_models = []
		for model in self.models:
			self.fitted_models.append(model(**kwargs).fit(self.df[self.vars_of_interest],df[self.y]))

    def plot_against_y(self,function=None):
        '''Where colour is squared error or some other var'''
        #do linked plots here
        cat, cont, time = cat_cont_time(self.df[self.vars_of_interest])
#        cat = self.df.columns[self.df.dtypes=='category']
#        cont =  self.df.columns[self.df.dtypes=='float64']
        #first continuous
        fig, axs = plt.subplots(nrows=1,ncols=len(cont),sharey=True)
        for ax,col in zip(axs.flat, cont):
            sns.regplot(x=col,y=self.y,data=self.df,ax=ax)
#        g = sns.lmplot(x="total_bill", y=self.y, data=self.df)
        #then categorical
        
        fig, axs = plt.subplots(nrows=1,ncols=len(cat),sharey=True)
        for ax,col in zip(axs.flat, cat):
            sns.violinplot(x=col,y=self.y,data=self.df,ax=ax)
            
#        g = sns.FacetGrid(self.df,col=self.df.columns[self.df.dtypes=='category'],row=self.y,sharey=True)
#        g.map(sns.violinplot)
        return fig
        
    def linreg(self):
        '''Uses statsmodels OLS for linear regression. Automatically inserts constant'''
        lm = sm.OLS(endog=ytrain,exog=add_const(self.df[self.vars_of_interest]),hasconst=1).fit()
        print lm.summary2()
        
    def plot_residuals(self):
        self.log.append('plot_residuals')
        #sns.residplot 
#        https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.residplot.html
    
    def heteroscedacity_check(self):
        self.log.append('')

    def anova(self):
        self.log.append('anova')



        

if __name__ == '__main__':
#    from sklearn.datasets import make_classification
#
#    x, y = make_classification(n_features=10, n_samples=1000, n_informative=5,
#                               n_clusters_per_class=3, n_redundant=0, hypercube=True, flip_y=.5)
#    x = pd.DataFrame(x)
#    bc = Classification(x,y)
##    l = pd.read_csv('../deprivationProject/licwData.csv')
##    l = l.drop(u'Unnamed: 0', axis=1)
##    l = l.drop(0, axis=0)
#    d = pd.read_csv('../deprivationProject/deprivationColumn').True
#    c = Classification(l, d)
#    bc.reduce_dimensions()
#    bc.plot_kdes()
#    bc.plot_predictions()

    from bokeh.sampledata.autompg import autompg
    a=Regression(autompg,'mpg')

    titanic = sns.load_dataset("titanic")
    t=Regression(titanic,'fare')
    
    iris = sns.load_dataset("iris")
#    sns.pairplot(iris)
