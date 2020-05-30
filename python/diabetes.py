# from sklearn.datasets import dump_svmlight_file
# from sklearn.datasets import load_diabetes
# import scipy.io
# import numpy as np
#
# mat = scipy.io.loadmat('diabetes.mat')
# #dump_svmlight_file(mat['Xtrain'], np.reshape(mat['Ytrain'],(-1)), 'shuttle_train')
# #dump_svmlight_file(mat['Xtest'], np.reshape(mat['Ytest'],(-1)), 'shuttle_test')
#
# x_train = mat['Xtrain']
# y_train = mat['Ytrain']
# x_test = mat['Xtest']
# y_test = mat['Ytest']
#Importing basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importing the Dataset
dataset = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(np.vstack((x_train,x_test)),
#                                                     np.vstack((y_train,y_test)),
#                                                     test_size = 0.15,
#                                                     random_state = 45)

#Splitting the data into dependent and independent variables
Y = dataset.Outcome
x = dataset.drop('Outcome', axis = 1)
columns = x.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
data_x = pd.DataFrame(X, columns = columns)

#Splitting the data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.15, random_state = 45)

from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)
print(np.bincount(y_train))

from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy of Random Forest on test set: {:.2f}'.format(model.score(x_test, y_test)))
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))