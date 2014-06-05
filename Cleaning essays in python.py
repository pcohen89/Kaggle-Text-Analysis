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
import statsmodels as sm
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

# create a word count variable

train['word_count'] = [len(cell) for cell in train.essay]
test['word_count'] = [len(cell) for cell in test.essay]

#use a vectorizer to count word usage instances

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit(train.essay)
train_tokens = vectorizer.transform(train.essay)
test_tokens = vectorizer.transform(test.essay)


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

# run ridge with optimal penalization

t0= time.time()
ridge= RidgeCV(alphas=optimizer.x, store_cv_values=True, normalize=True)
ridge.fit(train_tokens[0:documents], train.is_exciting[0:documents])
print "It took {time} minutes to run the optimized ridge".format(time=(time.time()-t0)/60)  
predictions = ridge.predict(train_tokens)  
rmse = np.sqrt(np.mean((train.is_exciting-predictions)**2))
rmse

# create an OLS regression for word count
ols= sm.regression.linear_model.OLS(train.is_exciting, train.word_count)
results= ols.fit()

# add ols and ridge predictions to train and test

train['ridge_predictions']=ridge.predict(train_tokens) 
train['length_predictions'] = train.word_count*results.params[0]
test['ridge_predictions']=ridge.predict(test_tokens) 
test['length_predictions'] = test.word_count*results.params[0]

data_for_ensemble['ridge_predictions'] = pd.DataFrame(train.ridge_predictions)
data_for_ensemble['length_predictions'] = pd.DataFrame(train.length_predictions)

# create a ridge regression that incorporates the bag of words and essay length
ridge= RidgeCV(alphas=1, store_cv_values=True, normalize=True)
ridge.fit(train.ridge_predictions, train.is_exciting)
print "It took {time} minutes to run the optimized ridge".format(time=(


#create an entry
entry = pd.DataFrame(data=test['projectid']) 
entry['is_exciting'] = ridge.predict(test_tokens)
entry.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Submissions/6.5.2014 PC submission.csv", index=False)


# extra code snippets
##
def test_function(alpha):
    return ((0.2*alpha)*(0.2*alpha)+3*alpha+7)
    
optimizer = minimize(test_function, [0,1], method='nelder-mead', options= {'xtol':1e-2, 'disp':True})

 
exes = [0,1]
    
penalties = np.arange(1 , 4, .1)
##



