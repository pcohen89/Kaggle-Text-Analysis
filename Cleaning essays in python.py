# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:07:35 2014

@author: p_cohen
"""

import pandas as pd
import re
import statsmodels.api as sts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.ridge import RidgeCV
import numpy as np
import time
import statsmodels.api as sm
from scipy.optimize import minimize


essays = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\essays.csv")
essays.count()

# clean control characters out of data

for column in essays.columns:
    essays[column]= [str(cell) for cell in essays[column]]
    essays[column]= [re.sub('\W', ' ', cell) for cell in essays[column]]
    
# bring in outcomes data
    
outcomes= pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\outcomes.csv")
outcomes.count()

# recode is_exciting

outcomes.is_exciting[outcomes.is_exciting=='t'] = 1
outcomes.is_exciting[outcomes.is_exciting=='f'] = 0

# merge with essays

essays_w_outcomes = essays.merge(outcomes, on='projectid')
#Note we only have outcomes for a subset of the essays 619000/640000
print(essays_w_outcomes['projectid'].count())

#keep only where essay and outcomes is filled (unused)
index_nomiss_exciting = pd.notnull(essays_w_outcomes.is_exciting) 
index_nomiss_essay = pd.notnull(essays_w_outcomes.essay)
essays_w_outcomes = essays_w_outcomes[index_nomiss_essay & index_nomiss_exciting]
essays_w_outcomes[index_nomiss_essay & index_nomiss_exciting]

#Create some basic variables that could be interesting

essays_w_outcomes['essay_len'] = [len(cell) for cell in essays_w_outcomes.essay]
essays_w_outcomes['is_essay_long'] = [len(cell)>2000 for cell in essays_w_outcomes.essay]

#use a vectorizer to count word usage instances

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(essays_w_outcomes.essay)

# when we are really ready to do this seriously, fit transformation on train only
# then transform the whole thing


# use optimizer to find best penalty for ridge

def pc_ridge(penalty):
    ridge= RidgeCV(alphas= penalty, store_cv_values=True, normalize=True)
    ridge.fit(X[0:documents], essays_w_outcomes.is_exciting[0:documents])
    predictions = ridge.predict(X)
    rmse = np.sqrt(np.mean((essays_w_outcomes.is_exciting-predictions)**2))
    return rmse
    
documents = 20000

t0= time.time()
optimizer = minimize(pc_ridge, array([15]), method='nelder-mead', options= {'xtol':1e-2, 'disp':True})
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)


ridge= RidgeCV(alphas=optimizer.x, store_cv_values=True, normalize=True)
ridge.fit(X[0:documents], essays_w_outcomes.is_exciting[0:documents])  
predictions = ridge.predict(X)  
rmse = np.sqrt(np.mean((essays_w_outcomes.is_exciting-predictions)**2))
rmse





# extra code snippets
##
def test_function(alpha):
    return ((0.2*alpha)*(0.2*alpha)+3*alpha+7)
    
optimizer = minimize(test_function, [0,1], method='nelder-mead', options= {'xtol':1e-2, 'disp':True})

 
exes = [0,1]
    
penalties = np.arange(1 , 4, .1)
##



