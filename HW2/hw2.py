import os
import pandas as pd
import numpy as np

'''
Read Input Data
'''
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_size = len(df)
test_size = len(df_test)

'''
Data Preprocessing
'''
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

imputer = SimpleImputer(strategy='median')
df = imputer.fit_transform(df)
df_test = imputer.fit_transform(df_test)

y_resampled = df[:,-1]
x_resampled = df[:,:-1]
# x_train = df.drop(df.columns[-1], axis=1)


# x = x_train.values
# y = y_train.values
# print(x_train.iloc[[0, 1]])

# rus = RandomUnderSampler(random_state=42)
# x_resampled, y_resampled = rus.fit_resample(x_train, y_train)

x_test = df_test

sc = StandardScaler()
sc.fit(x_resampled)
x_resampled = sc.transform(x_resampled)
x_test = sc.transform(x_test)

# pca = PCA(n_components=15)
# x_resampled = pca.fit_transform(x_resampled)
# x_test = pca.transform(x_test)

# print(x_resampled)
# print(y_resampled)

'''
Train Model
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mode

# RandomForest
# '''
rfc = RandomForestClassifier(n_estimators = 110 , n_jobs = 4, random_state=10)
rfc.fit(x_resampled , y_resampled)
rfc_pred = rfc.predict(x_test)
# rfc = RandomForestClassifier(n_estimators = 105 , n_jobs = 4)
# rfc.fit(x_resampled , y_resampled)
# rfc_pred = rfc.predict(x_test)
# '''

# RBF SVM 0.029
'''
svc = SVC(kernel='rbf' ,C=2 ,random_state=0) 
svc.fit(x_resampled, y_resampled)
svc_pred = svc.predict(x_test)
# '''

# Linear SVM  
'''
svc = SVC(kernel='linear' ,C=2 ,random_state=0)
svc.fit(x_resampled, y_resampled)
svc_pred = svc.predict(x_test)
'''

# Gaussian NB 0.47825
'''
gnb = GaussianNB()
gnb.fit(x_resampled, y_resampled)
gnb_pred = gnb.predict(x_test)
'''

'''
Output Predictions Data
'''
output = pd.DataFrame({'label': rfc_pred})
output.to_csv('myAns.csv', index_label='Id')

# output = pd.DataFrame({'label': svc_pred})
# output.to_csv('svc.csv', index_label='Id')

# output = pd.DataFrame({'label': gnb_pred})
# output.to_csv('gnb.csv', index_label='Id')

# Perform majority voting
# predictions = np.array([rfc_pred, svc_pred, gnb_pred])
# majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.round().astype('int64'))), axis=0, arr=predictions)
# predictions = np.where(np.all(predictions == majority_vote, axis=0), rfc_pred, majority_vote)

# # Save the predictions to a CSV file
# output = pd.DataFrame({'label': predictions})
# output.to_csv('myAns.csv', index_label='Id')