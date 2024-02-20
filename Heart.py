#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import PredefinedSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# This function displays the splits of the tree
from sklearn.tree import plot_tree

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


from xgboost import XGBClassifier

# This is the function that helps plot feature importance 
from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# This displays all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# This module lets us save our models once we fit them.
import pickle



df = pd.read_csv('C:/Users/ACER/Downloads/heart_2022_with_nans.csv.zip')

df1 = df.copy()

df1 = df1.dropna()

health_df = df1[['Sex', 'GeneralHealth', 'SleepHours','HadAngina', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadKidneyDisease',
               'HadArthritis', 'HadDiabetes', 'SmokerStatus','ECigaretteUsage','BMI', 
               'AlcoholDrinkers', 'HIVTesting', 'HighRiskLastYear', 'CovidPos', 'PhysicalActivities', 'FluVaxLast12', 'HadHeartAttack']]


health_df['SleepHours'] = health_df['SleepHours'].astype(int)

health_df['BMI'] = health_df['BMI'].astype(int)

activities = {
    'Yes': '1',
    'No': '0'
}

health_df['PhysicalActivities'] = health_df['PhysicalActivities'].replace(activities).astype(int)

Angina = {
    'Yes': '1',
    'No': '0'
}

health_df['HadAngina'] = health_df['HadAngina'].replace(Angina).astype(int)

Asthma = {
    'Yes': '1',
    'No': '0'
}

health_df['HadAsthma'] = health_df['HadAsthma'].replace(Asthma).astype(int)

Diabetes = {
    'Yes': '1',
    'No': '0',
    'No, pre-diabetes or borderline diabetes' : '0',
    'Yes, but only during pregnancy (female)' : '1'
}

health_df['HadDiabetes'] = health_df['HadDiabetes'].replace(Diabetes).astype(int)



alcohol = {
    'Yes': '1',
    'No': '0',
}

health_df['AlcoholDrinkers'] = health_df['AlcoholDrinkers'].replace(alcohol).astype(int)

HIV = {
    'Yes': '1',
    'No': '0',
}

health_df['HIVTesting'] = health_df['HIVTesting'].replace(HIV).astype(int)

FluVaxLast12 = {
    'Yes': '1',
    'No': '0',
}

health_df['FluVaxLast12'] = health_df['FluVaxLast12'].replace(HIV).astype(int)


HighRiskLastYear = {
    'Yes': '1',
    'No': '0',
}

health_df['HighRiskLastYear'] = health_df['HighRiskLastYear'].replace(HighRiskLastYear).astype(int)

HadSkinCancer = {
    'Yes': '1',
    'No': '0',
}

health_df['HadSkinCancer'] = health_df['HadSkinCancer'].replace(HadSkinCancer).astype(int)

HadCOPD = {
    'Yes': '1',
    'No': '0',
}

health_df['HadCOPD'] = health_df['HadCOPD'].replace(HadCOPD).astype(int)


HadArthritis = {
    'Yes': '1',
    'No': '0',
}

health_df['HadArthritis'] = health_df['HadArthritis'].replace(HadArthritis).astype(int)

health_dict = {
    'Excellent': 5,
    'Very good': 4,
    'Good': 3,
    'Fair':2,
    'Poor':1
}

health_df['GeneralHealth'] = health_df['GeneralHealth'].replace(health_dict).astype(int)

heart_dict = {
    'Yes': '1',
    'No': '0'
}

health_df['HadHeartAttack'] = health_df['HadHeartAttack'].replace(heart_dict).astype(int)

HadKidneyDisease = {
    'Yes': '1',
    'No': '0',
}

health_df['HadKidneyDisease'] = health_df['HadKidneyDisease'].replace(HadKidneyDisease).astype(int)


health_df['Sex'] = health_df['Sex'].map({'Male': 1, 'Female': 0})

health_df['SmokerStatus'] = health_df['SmokerStatus'].map({'Former smoker': 1, 'Current smoker - now smokes every day': 3,'Current smoker - now smokes some days':2, 'Never smoked': 0}) 

health_df['ECigaretteUsage'] = health_df['ECigaretteUsage'].map({'Never used e-cigarettes in my entire life':0, 'Use them some days': 1, 'Not at all (right now)': 0,'Use them every day':2})

health_df['CovidPos'] = health_df['CovidPos'].map({'Yes':1, 'No': 0, 'Tested positive using home test without a health professional':1})
health_df['CovidPos'] = health_df['CovidPos'].astype(int)


# In[2]:


health_df = health_df.dropna()

health_df.isna().sum()


# In[3]:


# Define the y (target) variable
y = health_df['HadHeartAttack']

# Define the X (predictor) variables
X = health_df.copy()
X = X.drop('HadHeartAttack', axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, stratify=y, 
                                                    random_state=42)


# In[6]:


xgb = XGBClassifier(objective='binary:logistic', random_state=0) 

cv_params = {'max_depth': [4,5], 
             'min_child_weight': [1,2,3],
             'learning_rate': [0.1, 0.2],
             'n_estimators': [75]
             }    

scoring = {'accuracy', 'precision', 'recall', 'f1'}

xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='f1')

#%%time
xgb_cv.fit(X_train, y_train)


# In[8]:

# change this file path to yours or serve flow chart
path = 'C:/Users/ACER/Documents/Python Scripts'

# Pickle the model
with open(path + 'xgb_cv_model.pickle', 'wb') as to_write:
    pickle.dump(xgb_cv, to_write) 

# Open pickled model
with open(path+'xgb_cv_model.pickle', 'rb') as to_read:
    xgb_cv = pickle.load(to_read)



# In[9]:


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.
  
    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.  
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                         }
                        )
  
    return table

# Create xgb model results table
xgb_cv_results = make_results('XGBoost CV', xgb_cv)
xgb_cv_results


# In[14]:


xgb_cv_preds = xgb_cv.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




