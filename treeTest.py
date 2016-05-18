# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:15:03 2015

@author: d
"""

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, tree, _tree, _utils,export_graphviz,export
import seaborn as sns
from bokeh.sampledata.autompg import autompg
from pprint import pprint
import pandas as pd

titanic = sns.load_dataset("titanic")
y=titanic.pop('survived')
d=DecisionTreeClassifier()
X=titanic[[u'pclass', u'age', u'sibsp', u'parch', u'fare',
         u'adult_male',
       u'alone']]
#d.fit(X,y)

y=autompg['mpg']
X=autompg.drop(['name','mpg'],axis=1)
r=DecisionTreeRegressor(min_samples_leaf=25)
r.fit(X,y)
from sklearn.externals.six import StringIO
with open("mpg.dot", 'w') as f:
     f = export_graphviz(r, out_file=f, feature_names=X.columns)


pprint(zip(X.columns[r.tree_.feature],r.tree_.threshold,r.tree_.children_left,r.tree_.children_right,r.tree_.value))

#import os
#os.unlink('iris.dot')
#def tree_to_dict(tree)
d={}
d['feature']=X.columns[r.tree_.feature]
d['threshold']=r.tree_.threshold
d['left_children']=r.tree_.children_left
d['right_children']=r.tree_.children_right
d['value']=r.tree_.value.flatten()
d['impurity']=r.tree_.impurity
tree_df=pd.DataFrame(d)
print tree_df