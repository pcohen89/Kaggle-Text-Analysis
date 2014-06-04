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

# merge with essays to create train data

unsplit = essays.merge(outcomes, how='outer', on='projectid')
print(unsplit['projectid'].count())
test = unsplit[pd.isnull(unsplit.is_exciting)]
train = unsplit[pd.notnull(unsplit.is_exciting)]

#use a vectorizer to count word usage instances

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit(train.essay)
train_tokens = vectorizer.transform(train.essay)
test_tokens = vectorizer.transform(test.essay)


# when we are really ready to do this seriously, fit transformation on train only
# then transform the whole thing


# use optimizer to find best penalty for ridge

def pc_ridge(penalty):
    ridge= RidgeCV(alphas= penalty, store_cv_values=True, normalize=True)
    ridge.fit(train_tokens[0:documents], train.is_exciting[0:documents])
    predictions = ridge.predict(train_tokens)
    rmse = np.sqrt(np.mean((train.is_exciting-predictions)**2))
    return rmse
    
documents = 25000

t0= time.time()
optimizer = minimize(pc_ridge, array([15]), method='nelder-mead', options= {'xtol':1e-2, 'disp':True})
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)

t0= time.time()
ridge= RidgeCV(alphas=optimizer.x, store_cv_values=True, normalize=True)
ridge.fit(train_tokens[0:documents], train.is_exciting[0:documents])
print "It took {time} minutes to run the optimized ridge".format(time=(time.time()-t0)/60)  
predictions = ridge.predict(train_tokens)  
rmse = np.sqrt(np.mean((train.is_exciting-predictions)**2))
rmse

entry = pd.DataFrame(data=test['projectid']) 
entry['predictions'] = ridge.predict(test_tokens)



# extra code snippets
##
def test_function(alpha):
    return ((0.2*alpha)*(0.2*alpha)+3*alpha+7)
    
optimizer = minimize(test_function, [0,1], method='nelder-mead', options= {'xtol':1e-2, 'disp':True})

 
exes = [0,1]
    
penalties = np.arange(1 , 4, .1)
##



