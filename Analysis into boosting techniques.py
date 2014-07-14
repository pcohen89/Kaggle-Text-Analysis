# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:40:43 2014

@author: p_cohen
"""
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import statsmodels as sm
from sklearn.metrics import roc_curve, auc
from scipy.optimize import minimize

outcomes= pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\outcomes.csv")
essays = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\essays.csv")
projects = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw\projects.csv")
resource = pd.read_csv("S:\General\Training\Ongoing Professional Development\Kaggle\Predicting Excitement at DonorsChoose\Data\Raw/resources.csv")

# clean control characters out of essay data

for column in essays.columns:
    essays[column]= [str(cell) for cell in essays[column]]
    essays[column]= [re.sub('\W', ' ', cell) for cell in essays[column]]
 
# create word count variables for the various text fields  in essay data
text_vars = ('title', 'short_description', 'need_statement', 'essay')
for var in text_vars:
    new_var = 'word_count_' + var 
    essays[new_var] = [len(cell) for cell in essays[var]]

# drop messy resource data 

resource_to_drop = ('vendor_name','resourceid', 'vendorid', 'item_name', 'item_number')
for var in resource_to_drop:
    del resource[var]
    
# create a variable for the sum of items requested per project
 
sum_item_quant= resource.groupby('projectid')['item_quantity'].sum()
resource= resource.merge(sum_item_quant.reset_index(), how='outer', on='projectid')

# drop resource duplicates on project id (THINK ABT HOW TO IMPROVE THIS)

resource = resource.drop_duplicates(cols='projectid')

# merge data together on projectid
unsplit = projects.merge(outcomes, how='outer', on='projectid')
unsplit = unsplit.merge(essays, how='outer', on='projectid')
unsplit = pd.merge(unsplit, resource, how='left', left_on='projectid', right_on='projectid')

#outer merge, test sample is projects where is_exciting is missing
test = unsplit[pd.isnull(unsplit.is_exciting)]
train_and_validation = unsplit[pd.notnull(unsplit.is_exciting)]
train_and_validation =train_and_validation[pd.to_datetime(train_and_validation.date_posted)>pd.to_datetime('2012-1-1')]
train_and_validation['is_exciting'][train_and_validation['is_exciting']=='f'] = -1
train_and_validation['is_exciting'][train_and_validation['is_exciting']=='t'] = 1 
non_test_obs = len(train_and_validation.is_exciting)

random_vector =array(np.random.rand((len(train_and_validation.is_exciting)),1))
# create a teacher level score variable

for_teacher_grouping = pd.DataFrame(train_and_validation.is_exciting[np.squeeze(random_vector>.2)].astype(int))
for_teacher_grouping['teacher_acctid_x'] = train_and_validation['teacher_acctid_x'][np.squeeze(random_vector>.2)]
for_teacher_grouping.rename(columns={'is_exciting': 'teacher_average'},
                            inplace=True)
grouped_teacher_mean =  for_teacher_grouping.groupby('teacher_acctid_x').mean()
grouped_teacher_count =  for_teacher_grouping.groupby('teacher_acctid_x').teacher_average.count()
pd_grpd_teach_mn = grouped_teacher_mean.reset_index()
pd_grpd_teach_ct = grouped_teacher_count.reset_index()
pd_grpd_teach_ct.rename(columns={0: 'count_per_teacher'},
                        inplace=True)
grpd_vars = pd_grpd_teach_ct.merge(pd_grpd_teach_mn, how='outer', 
                                   on='teacher_acctid_x')
grpd_vars['teacher_w_freqwt'] = grpd_vars.count_per_teacher*grpd_vars.teacher_average                       

train_and_validation = train_and_validation.merge(grpd_vars,
                                                  how='outer', on='teacher_acctid_x')
test = test.merge(grpd_vars,
                how='left', on='teacher_acctid_x')                       

# add predictions from bag of words ridge to data
def bag_of_words_ridge(variable):
    vectorizer = TfidfVectorizer(min_df=.1, max_df=.9) #use a vectorizer to count word usage instances and create sparse matrix
    bag_of_words_X = vectorizer.fit(train_and_validation[variable][pd.to_datetime(train_and_validation.date_posted)>pd.to_datetime('2013-11-1')])
    # normalization of vectorizer is fit using train only
    bag_of_words_X = vectorizer.transform(train_and_validation[variable])
    test_bag_of_words= vectorizer.transform(test[variable])
    ridge= RidgeCV(array([18]), store_cv_values=True, normalize=True)
    # using data range to gaurantee recency and also run time 
    ridge.fit(bag_of_words_X[pd.to_datetime(train_and_validation.date_posted)>pd.to_datetime('2013-11-8')], train_and_validation.is_exciting[pd.to_datetime(train_and_validation.date_posted)>pd.to_datetime('2013-11-8')])
    var_nm = "b_of_wds_prds_" + variable
    # put predictions into samples for use later as base classifiers in ada boost    
    train_and_validation[var_nm]=ridge.predict(bag_of_words_X)
    test[var_nm]=ridge.predict(test_bag_of_words)
   
#initialize the text fields in essays
text_vars = ('title', 'essay', 'short_description' )

for var in text_vars:
    t0 = time.time()
    bag_of_words_ridge(var)
    print "For " + var 
    print "It took {timer} minutes to run the optimized ridge".format(timer=(time.time()-t0)/60)
 
# export data

train_and_validation.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Clean/7.6.2014 train_and_validation.csv", index=False)
test.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Clean/7.6.2014 test.csv", index=False)

# keep certain vars for  modeling
x_vars = ('students_reached',
'total_price_excluding_optional_support', 'school_latitude',
'school_longitude', 'is_exciting', 
'item_quantity_y', 'item_unit_price',
# word count vars
'word_count_title','word_count_essay','word_count_short_description', 'word_count_need_statement',
# bag of words regression predictions from various text features
'b_of_wds_prds_title', 'b_of_wds_prds_essay', 'b_of_wds_prds_short_description')

possible_vars =  ('primary_focus_subject', 'secondary_focus_subject', 'project_resource_type')

t_f_vars = ('school_charter', 'eligible_double_your_impact_match', 'eligible_almost_home_match',
'school_magnet','teacher_teach_for_america', 'teacher_ny_teaching_fellow', 
'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise' )

level_vars = ('poverty_level', 'school_kipp', 'school_state', 'teacher_prefix' ,
                    'primary_focus_subject', 'primary_focus_area', 
                     'secondary_focus_subject', 'secondary_focus_area', 'grade_level',
                    'resource_type')  
 
#level_vars = ()
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
    for level in levels:
        new_var_nm = "is_" + str(object=level_var) + "_" + str(object=level)
        output_data[new_var_nm ]=0
        trues= input_data[level_var]==level
        output_data[new_var_nm][trues] = 1   
        
#create a data set that will use the good modeling vars from the wide data
train_X = pd.DataFrame(train_and_validation.fulfillment_labor_materials)
test_X = pd.DataFrame(test.fulfillment_labor_materials)
#code vars that are copies of vars in full data
for vars in x_vars:
    train_X[vars] = train_and_validation[vars] 
    test_X[vars] = test[vars] 
#recode missings to zero
train_X.fillna(0, inplace=True)
test_X.fillna(0, inplace=True)
# create is_exciting score vars for many levelled vars
for level_var in level_vars:    
    create_score_variables(level_var, train_X,train_and_validation)
    create_score_variables(level_var, test_X,test)

code_tf_vars(train_and_validation, train_X)
code_tf_vars(test, test_X)

features= len(train_X.columns)  
    
del test_X['is_exciting']

# split train and validation

train_features = pd.DataFrame(train_X[random_vector>.2])
train_outcome = train_features.is_exciting
del train_features['is_exciting']
validation = pd.DataFrame(train_X[random_vector<=.2])
validation_for_p =pd.DataFrame(train_X[random_vector<=.2])
del validation_for_p['is_exciting']

# Try a bunch of different boosting techniques
def run_adaboost(estimators_and_learn_rt):
    print estimators_and_learn_rt[0]
    print estimators_and_learn_rt[1]
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, 
                                                                   max_features=features-1,
                                                                   splitter='best', min_samples_leaf=10),
                                                                   n_estimators = int(estimators_and_learn_rt[0]), 
                                                                   learning_rate=estimators_and_learn_rt[1])
    clf.fit(train_features, train_outcome)
    validation['predictions_clf']=clf.predict_proba(validation_for_p)[:,1]
    fpr, tpr, thresholds = roc_curve(validation.is_exciting, validation.predictions_clf)
    auc_score = auc(fpr,tpr)
    return auc_score
   
# we run an optimizer to find the penalty that minimizes rmse of ridge
init_guess =array([1200, .05])
# init_guess initializes the opimization with a guess of the optimal penalty 
   
t0= time.time()
optimizer = minimize(run_adaboost, init_guess, method='nelder-mead', options= {'xtol':5e-4, 'disp':True})
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)

t0= time.time()
run_adaboost(array([1500,.005]))
print "It took {time} minutes to optimize".format(time=(time.time()-t0)/60)

# add variables that have hurt adaboost performance

extra_level_vars = ( 'school_state', 'teacher_prefix' ,
                    'primary_focus_subject', 'primary_focus_area', 
                     'secondary_focus_subject', 'secondary_focus_area', 'grade_level',
                    'resource_type') 

for level_var in extra_level_vars:    
    create_score_variables(level_var, train_features,train_and_validation[random_vector>.2])
    create_score_variables(level_var, validation,train_and_validation[random_vector<=.2])
    create_score_variables(level_var, validation_for_p,train_and_validation[random_vector<=.2])
    create_score_variables(level_var, test_X, test)

# add in teacher average
def teacher_mean_forms(x):
    name = 'teacher_w_freqwt_' + str(object=x)
    train_features[name] = pd.DataFrame((train_and_validation.teacher_w_freqwt[np.squeeze(random_vector>.2)])**x)
    validation_for_p[name] = pd.DataFrame((train_and_validation.teacher_w_freqwt[np.squeeze(random_vector>.2)])**x)
    test_X[name] = pd.DataFrame((test.teacher_w_freqwt)**x)

forms = (.5,2,1)

for x in forms:
    teacher_mean_forms(x)

# add in a month

def make_month_var(input_data, output_data, sample_split_conditional):
    trn_feats_dates = pd.DatetimeIndex(input_data.date_posted[sample_split_conditional])
    output_data['date_for_mod'] = trn_feats_dates.astype(int64) //10**9
    #output_data['month'] = month(output_data['date_posted']
one_vector =array(np.random.rand((len(test.is_exciting)),1))  
make_month_var( train_and_validation, train_features, np.squeeze(random_vector>.2) )
make_month_var( train_and_validation, validation_for_p, np.squeeze(random_vector<=.2) )
make_month_var(  test, test_X, np.squeeze(one_vector>-1) )

# create a logistic model with the adaboost and extra vars
train_features.fillna(0, inplace=True)
validation_for_p.fillna(0, inplace=True)
test_X.fillna(0, inplace=True)

# run random forests
t0= time.time()
rndm_forest_clf = RandomForestClassifier(n_estimators=600, min_samples_split=10, min_samples_leaf=3)
rndm_forest_clf.fit(train_features, train_outcome)
validation['predictions_forest_clf']=rndm_forest_clf.predict_proba(validation_for_p)[:,1]
fpr, tpr, thresholds = roc_curve(validation.is_exciting, validation.predictions_forest_clf)
auc(fpr,tpr)
print "It took {time} minutes to run forests".format(time=(time.time()-t0)/60)
forest_features = len(train_features.columns)

#run logistic
logit = LogisticRegression()
logit.fit(train_features, train_outcome)
logit_feats = len(train_features.columns)

validation['predictions']=logit.predict_proba(validation_for_p)[:,1]
fpr, tpr, thresholds = roc_curve(validation.is_exciting, validation.predictions)
auc_score = auc(fpr,tpr)
auc_score 

# run ridge

full_ridge= RidgeCV(np.array([7]), store_cv_values=True, normalize=True)
# using data range to gaurantee recency and also run time 
full_ridge.fit(train_features, train_outcome)
validation['predictions']=logit.predict_proba(validation_for_p)[:,1]
fpr, tpr, thresholds = roc_curve(validation.is_exciting, validation.predictions)
auc_score = auc(fpr,tpr)
auc_score 
  
    
# add predictions to train features
ens_train_features['Adaboost'] = pd.DataFrame(clf.predict_proba(train_features.iloc[:,0:30])[:,1])
validation_for_p['Adaboost'] = pd.DataFrame(clf.predict_proba(validation_for_p.iloc[:,0:30])[:,1])
test_X['Adaboost'] = pd.DataFrame(clf.predict_proba(test_X.iloc[:,0:30])[:,1])

ens_train_features['Forest'] = rndm_forest_clf.predict_proba(train_features.iloc[:,0:forest_features])[:,1]
validation_for_p['Forest'] = rndm_forest_clf.predict_proba(validation_for_p.iloc[:,0:forest_features])[:,1]
test_X['Forest'] = rndm_forest_clf.predict_proba(test_X.iloc[:,0:forest_features])[:,1]


ens_train_features['Logit'] = logit.predict_proba(train_features.iloc[:,0:logit_feats])[:,1]
validation_for_p['Logit'] = logit.predict_proba(validation_for_p.iloc[:,0:logit_feats])[:,1]
test_X['Logit'] = logit.predict_proba(test_X.iloc[:,0:logit_feats])[:,1]

ens_train_features['Ridge'] = full_ridge.predict(train_features.iloc[:,0:logit_feats])[:,1]
validation_for_p['Ridge'] = full_ridge.predict(validation_for_p.iloc[:,0:logit_feats])[:,1]
test_X['Ridge'] = full_ridge.predict(test_X.iloc[:,0:logit_feats])[:,1]

# final tree

ens_forest_clf = RandomForestClassifier(n_estimators=600, min_samples_split=6, min_samples_leaf=2)
ens_forest_clf.fit(ens_train_features, train_outcome)


validation['predictions']=ens_forest_clf.predict_proba(validation_for_p.iloc[:,167:171])[:,1]
fpr, tpr, thresholds = roc_curve(validation.is_exciting, validation.predictions)
auc_score = auc(fpr,tpr)
auc_score 

# submission
entry = pd.DataFrame(data=test['projectid']) 
entry['is_exciting'] =ens_logit.predict_proba(test_X.iloc[:,167:171])[:,1]
#entry['is_exciting'] = test.length_predictions 
entry.to_csv("S:/General/Training/Ongoing Professional Development/Kaggle/Predicting Excitement at DonorsChoose/Data/Submissions/7.8.2014 many model ensemble .csv", index=False)



                                                       