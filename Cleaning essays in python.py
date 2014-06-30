# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:07:35 2014

@author: p_cohen
"""

import pandas as pd
import re
import statsmodels.api as sts
from sklearn.feature_extraction.text import TfidfVectorizer
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

# recode is_exciting to binary

outcomes.is_exciting[outcomes.is_exciting=='t'] = 1
outcomes.is_exciting[outcomes.is_exciting=='f'] = 0

# bring in dates from project data

projects= pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\projects.csv")
dates = pd.DataFrame(projects.projectid)
dates['date_posted'] = projects.date_posted

# merge with essays to create train data

unsplit = essays.merge(outcomes, how='outer', on='projectid')
unsplit = unsplit.merge(dates, how='left',on='projectid')
#outer merge, test sample is projects where is_exciting is missing
print(unsplit['projectid'].count())
test = unsplit[pd.isnull(unsplit.is_exciting)]
train = unsplit[pd.notnull(unsplit.is_exciting)]

# create a word count variable (actually character count)

train['word_count'] = [len(cell) for cell in train.essay]
test['word_count'] = [len(cell) for cell in test.essay]

# drop training data before 2013. This is clearly not optimal
train = train[train.date_posted>'2013-1-1']

#use a vectorizer to count word usage instances and create sparse matrix

vectorizer = TfidfVectorizer(min_df=.1, max_df=.9)
X = vectorizer.fit(train.essay)
# normalization of vectorizer is fit using train only
train_tokens = vectorizer.transform(train.essay)
test_tokens = vectorizer.transform(test.essay)

# use optimizer to find best penalty for ridge

documents = train.projectid.count()-1
#documents represents the number of essays to fit using. limited to this number
#due to processing time constraints

def pc_ridge(penalty): 
    # this function takes a complexity penalty as an input amd outputs RMSE
    ridge= RidgeCV(alphas= penalty, store_cv_values=True, normalize=True)
    ridge.fit(train_tokens[0:documents], train.is_exciting[0:documents])
    predictions = ridge.predict(train_tokens)
    return np.sqrt(np.mean((train.is_exciting-predictions)**2))
     # this is rmse 
    
    
def ensemble_ridge(penalty):
    
    ridge= RidgeCV(alphas=penalty, store_cv_values=True, normalize=True)
    ridge.fit(data_for_ensemble, train.is_exciting)
    predictions = ridge.predict(data_for_ensemble)
    return np.sqrt(np.mean((train.is_exciting-predictions)**2))
   
   
# we run an optimizer to find the penalty that minimizes rmse of ridge
init_guess = array([35])  
# init_guess initializes the opimization with a guess of the optimal penalty 
   
t0= time.time()
optimizer = minimize(pc_ridge, init_guess, method='nelder-mead', options= {'xtol':1e-2, 'disp':True})
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)

# run ridge with optimal penalization

t0= time.time()
ridge= RidgeCV(alphas=optimizer.x, store_cv_values=True, normalize=True)
# optimizer.x is the ridge penalty that minimized rmse
ridge.fit(train_tokens[0:documents], train.is_exciting[0:documents])
print "It took {time} minutes to run the optimized ridge".format(time=(time.time()-t0)/60)

# create an OLS regression for word count
ols= sm.regression.linear_model.OLS(train.is_exciting, train.word_count)
results= ols.fit()

# add ols and ridge predictions to train and test data 

train['ridge_predictions']=ridge.predict(train_tokens) 
train['length_predictions'] = train.word_count*results.params[0]
test['ridge_predictions']=ridge.predict(test_tokens) 
test['length_predictions'] = test.word_count*results.params[0]

data_for_ensemble = pd.DataFrame({"length_predictions":train.length_predictions,"ridge_predictions":train.ridge_predictions})

# create a ridge regression that incorporates the bag of words and essay length
init_guess_ens = array([.0125])   
t0= time.time()
ensemble_optimizer = minimize(ensemble_ridge, init_guess_ens, method='nelder-mead', options= {'xtol':1e-2, 'disp':True})
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)

ridge= RidgeCV(alphas=array([ensemble_optimizer.x]), store_cv_values=True, normalize=True)
ensemble = ridge.fit(data_for_ensemble, train.is_exciting)

#create a submission entry
entry = pd.DataFrame(data=test['projectid']) 
entry['is_exciting'] = ensemble.predict(test.iloc[:,19:21])
#entry['is_exciting'] = test.length_predictions 
entry.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Submissions/6.10.2014 PC submission .csv", index=False)



