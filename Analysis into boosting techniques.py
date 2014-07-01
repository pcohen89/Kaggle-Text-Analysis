# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:40:43 2014

@author: p_cohen
"""
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import pandas as pd
import re
import statsmodels.api as sts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.ridge import RidgeCV
import numpy as np
import time
import statsmodels as sm
from scipy.optimize import minimize

outcomes= pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\outcomes.csv")
outcomes.count()

essays = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\essays.csv")

essays['word_count'] = [len(cell) for cell in essays.essay]

# clean control characters out of data

for column in essays.columns:
    essays[column]= [str(cell) for cell in essays[column]]
    essays[column]= [re.sub('\W', ' ', cell) for cell in essays[column]]

projects = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\projects.csv")
unsplit = projects.merge(outcomes, how='outer', on='projectid')
unsplit = unsplit.merge(essays, how='outer', on='projectid')

#outer merge, test sample is projects where is_exciting is missing
print(unsplit['projectid'].count())
test = unsplit[pd.isnull(unsplit.is_exciting)]
train_and_validation = unsplit[pd.notnull(unsplit.is_exciting)]
train_and_validation  =train_and_validation[train_and_validation.date_posted>'2012-1-1']
train_and_validation['is_exciting'][train_and_validation['is_exciting']=='f'] = 0
train_and_validation['is_exciting'][train_and_validation['is_exciting']=='t'] = 1 


# add predictions from bag of words ridge to data

vectorizer = TfidfVectorizer(min_df=.1, max_df=.9) #use a vectorizer to count word usage instances and create sparse matrix
bag_of_words_X = vectorizer.fit(train_and_validation.essay)
# normalization of vectorizer is fit using train only
bag_of_words_X = vectorizer.transform(train_and_validation.essay)
test_bag_of_words= vectorizer.transform(test.essay)
documents = 25000
t0= time.time()
ridge= RidgeCV(array([18]), store_cv_values=True, normalize=True)
# optimizer.x is the ridge penalty that minimized rmse
ridge.fit(bag_of_words_X[0:documents], train_and_validation.is_exciting[0:documents])
print "It took {time} minutes to run the optimized ridge".format(time=(time.time()-t0)/60)

train_and_validation['bag_of_words_predictions']=ridge.predict(bag_of_words_X)
test['bag_of_words_predictions']=ridge.predict(test_bag_of_words)

x_vars = ('students_reached',
'total_price_excluding_optional_support', 'school_latitude',
'school_longitude', 'is_exciting', 'word_count', 'bag_of_words_predictions')

t_f_vars = ('school_county', 'school_charter', 'eligible_double_your_impact_match', 'eligible_almost_home_match',
'school_magnet','teacher_teach_for_america', 'teacher_ny_teaching_fellow', 
'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise')

level_vars = ('resource_type', 'grade_level', 'poverty_level', 
'primary_focus_area', 'primary_focus_subject', 'secondary_focus_subject', 'secondary_focus_area',
 'fulfillment_labor_materials','teacher_prefix' )  

# function that codes true false variables
def code_tf_vars(input_data, output_data):
    for var in t_f_vars:
        output_data[var]=0
        trues = input_data[var]=='t'
        output_data[var][trues] = 1 
    
# create and add grouped variables     
def create_score_variables(level_var, output_data, input_data):
    # this function creates a variable that represents the is_exciting probability
    # at each level of a categorical variable
    grouped = train_and_validation.groupby(level_var)   
    levels = grouped.groups.keys()
    new_var_nm = level_var + '_score'
    output_data[new_var_nm] = .999999
    for level in levels:
        resource_vector = train_and_validation[level_var]== level
        score = train_and_validation['is_exciting'][resource_vector].mean()
        output_vector = input_data[level_var]== level
        output_data[new_var_nm][output_vector] = score 
        
#create a data set that will use the good modeling vars from the wide data
train_X = pd.DataFrame(train_and_validation.fulfillment_labor_materials)

#code vars that are copies of vars in full data
for vars in x_vars:
    train_X[vars] = train_and_validation[vars] 
#recode missings to zero
train_X = train_X.fillna(0)

# create is_exciting score vars for many levelled vars
for level_var in level_vars:    
    create_score_variables(level_var, train_X,train_and_validation)

code_tf_vars(train_and_validation, train_X)
features= len(train_X.columns)

# code up test data identically

test_X = pd.DataFrame(test.fulfillment_labor_materials)
for vars in x_vars:
    test_X[vars] = test[vars] 
test_X = test_X.fillna(0)

code_tf_vars(test, test_X)
  
for level_var in level_vars:    
    create_score_variables(level_var, test_X,test)

del test_X['is_exciting']

# split train and validation
non_test_obs = len(train_and_validation.is_exciting)
train_features = pd.DataFrame(train_X.iloc[0:(.8*non_test_obs), :])

del train_features['is_exciting']
train_outcome = train_X.is_exciting[0:(.8*non_test_obs)]
validation = pd.DataFrame(train_X.iloc[(.8*non_test_obs+1):(non_test_obs-1),:])
validation_for_p =pd.DataFrame(train_X.iloc[(.8*non_test_obs+1):(non_test_obs-1),:])
del validation_for_p['is_exciting']

# Try a bunch of different boosting techniques
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, max_features=features-2, splitter='best'),n_estimators = 1400, learning_rate=.04)
clf.fit(train_features, train_outcome)
validation['predictions']=clf.predict_proba(validation_for_p)[:,1]
validation['predictions']= [round(cell,3) for cell in validation['predictions']]
tester= validation.groupby('predictions')
tester['is_exciting'].sum()/tester['is_exciting'].count()

# submission
entry = pd.DataFrame(data=test['projectid']) 
entry['is_exciting'] =clf.predict_proba(test_X)[:,1]
#entry['is_exciting'] = test.length_predictions 
entry.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Submissions/6.10.2014 PC submission ada pprob 2.csv", index=False)



                                                       