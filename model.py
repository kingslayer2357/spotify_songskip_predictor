# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:12:32 2021

@author: kingslayer
"""

#### JOINING CSV FILES TO PREPARE DATA ####


import pandas as pd

#Preparing Training Data

train_data=pd.read_csv(r'train_data.csv')
track_feats=pd.read_csv(r'track_feats.csv')


merged = train_data.merge(track_feats, left_on = ['track_id_clean'],
                right_on= ['track_id'],
                how = 'inner')
merged.to_csv(r'prepared_training_data.csv', index=False)


#Preparing Test Data


test_data=pd.read_csv(r'test_data.csv')


merged1 = test_data.merge(track_feats, left_on = ['track_id_clean'],
                right_on= ['track_id'],
                how = 'inner')
merged1.to_csv(r'prepared_testing_data.csv', index=False)


#Preparing Validation Data

validation_data=pd.read_csv(r'val_data.csv')

merged2 = validation_data.merge(track_feats, left_on = ['track_id_clean'],
                right_on= ['track_id'],
                how = 'inner')
merged2.to_csv(r'prepared_validation_data.csv', index=False)



#### DATA VISUALISATION AND FEATURE SELECTION ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#importing data

training_data=pd.read_csv(r'prepared_training_data.csv')
validation_data=pd.read_csv(r'prepared_validation_data.csv')


train_data=pd.DataFrame(training_data)
valid_data=pd.DataFrame(validation_data)


#Removing Unwanted or useless columns like session id etc.

train_data.info()


train_data.drop(columns=['session_id','track_id','track_id_clean'],inplace=True)
valid_data.drop(columns=['session_id','track_id','track_id_clean'],inplace=True)

#Making sure our data has no missing values

train_data.isna().any()
valid_data.isna().any()


#Visualisation

sns.countplot(train_data.not_skipped)

#FEATURE SELECTION

y_train=train_data.not_skipped
X_train=train_data.drop(columns=['not_skipped'])


y_valid=valid_data.not_skipped
X_valid=valid_data.drop(columns=['not_skipped'])




import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(fii.summary())


#Removing features on the basis of p-value
X_train.drop(columns=['session_length','bounciness','liveness','loudness'],inplace=True)
X_valid.drop(columns=['session_length','bounciness','liveness','loudness'],inplace=True)


import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(fii.summary())


X_train.drop(columns=['hist_user_behavior_reason_end_appload','key','acoustic_vector_4','acoustic_vector_7'],inplace=True)
X_valid.drop(columns=['hist_user_behavior_reason_end_appload','key','acoustic_vector_4','acoustic_vector_7'],inplace=True)


import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(fii.summary())


X_train.drop(columns=['hist_user_behavior_reason_end_backbtn','hist_user_behavior_reason_end_clickrow','hist_user_behavior_reason_end_endplay','hist_user_behavior_reason_end_fwdbtn','hist_user_behavior_reason_end_logout','hist_user_behavior_reason_end_remote'],inplace=True)
X_valid.drop(columns=['hist_user_behavior_reason_end_backbtn','hist_user_behavior_reason_end_clickrow','hist_user_behavior_reason_end_endplay','hist_user_behavior_reason_end_fwdbtn','hist_user_behavior_reason_end_logout','hist_user_behavior_reason_end_remote'],inplace=True)

import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(fii.summary())


X_train.drop(columns=['skip_1'],inplace=True)
X_valid.drop(columns=['skip_1'],inplace=True)

import statsmodels.api as sm
mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(fii.summary())



#### REGRESSION ####


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred_train=lr.predict(X_train)



from sklearn.metrics import accuracy_score

acc_train=accuracy_score(y_train,y_pred_train)
print('Training Accuracy:'+str(100*acc_train)+"%")   ## 99.30%

y_pred_valid=lr.predict(X_valid)
acc_valid=accuracy_score(y_valid,y_pred_valid)
print("Validation Accuracy:"+str(100*acc_valid)+"%")  ## 99.31%

#confusion matrix

from sklearn.metrics import confusion_matrix

cn=confusion_matrix(y_valid,y_pred_valid)
sns.heatmap(cn,annot=True)


#Saving our trained model
import pickle
# save the classifier
with open('my_spotify_classifier.pkl', 'wb') as fid:
    pickle.dump(lr, fid)   


#### TESTING OUR MODEL #####


#preparing data for testing

testing_data=pd.read_csv(r'prepared_testing_data.csv')
test_data=pd.DataFrame(testing_data)
test_data.drop(columns=['session_id','track_id','track_id_clean'],inplace=True)
y_test=test_data.not_skipped
X_test=test_data.drop(columns=['not_skipped'])


X_test.drop(columns=['session_length','bounciness','liveness','loudness','hist_user_behavior_reason_end_appload','key','acoustic_vector_4','acoustic_vector_7','hist_user_behavior_reason_end_backbtn','hist_user_behavior_reason_end_clickrow','hist_user_behavior_reason_end_endplay','hist_user_behavior_reason_end_fwdbtn','hist_user_behavior_reason_end_logout','hist_user_behavior_reason_end_remote','skip_1'],inplace=True)

#Loading our saved model

# and later you can load it
with open('my_spotify_classifier.pkl', 'rb') as f:
    lr = pickle.load(f)

y_pred_test=lr.predict(X_test)
acc_test=accuracy_score(y_test,y_pred_test)
print("Test Accuracy:"+str(100*acc_test)+"%")  ## 99.31%

#confusion matrix

from sklearn.metrics import confusion_matrix

cn_test=confusion_matrix(y_test,y_pred_test)
sns.heatmap(cn_test,annot=True)


#TEST RESULTS

test_results=pd.DataFrame()
test_results['True']=y_test
test_results['Predicted']=y_pred_test
test_results['Remark']=y_test-y_pred_test
test_results.Remark[(test_results['True']-test_results['Predicted']==0)]='Correct'
test_results.Remark[(test_results['True']-test_results['Predicted']!=0)]='Incorrect'

