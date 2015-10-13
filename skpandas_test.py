# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:57:49 2015

@author: Daniel D. Gibson
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn_pandas import DataFrameMapper, cross_val_score

data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                     'children': [4., 6, 3, 3, 2, 3, 5, 4],
                      'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})

mapper = DataFrameMapper([
    ('pet', sklearn.preprocessing.LabelBinarizer()),
    (['children'], sklearn.preprocessing.StandardScaler())
])



if __name__ == '__main__':
    pass
