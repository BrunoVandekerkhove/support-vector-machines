from sklearn.datasets import dump_svmlight_file
#from sklearn.datasets import load_diabetes
import scipy.io
import numpy as np

mat = scipy.io.loadmat('diabetes.mat')
#dump_svmlight_file(mat['Xtrain'], np.reshape(mat['Ytrain'],(-1)), 'shuttle_train')
#dump_svmlight_file(mat['Xtest'], np.reshape(mat['Ytest'],(-1)), 'shuttle_test')

x_train = mat['Xtrain']
y_train = mat['Ytrain']
x_test = mat['Xtest']
y_test = mat['Ytest']

from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
model.fit(x_train, y_train)
#y_pred = model.predict(mat['Xtest'])
print('Accuracy of Random Forest on test set: {:.2f}'.format(model.score(x_test, y_test)))