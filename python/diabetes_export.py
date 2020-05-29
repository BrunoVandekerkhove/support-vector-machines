from sklearn.datasets import dump_svmlight_file
#from sklearn.datasets import load_diabetes
import scipy.io
import numpy as np

mat = scipy.io.loadmat('shuttle.mat')
dump_svmlight_file(mat['Xtrain'], np.reshape(mat['Ytrain'],(-1)), 'shuttle_train')
dump_svmlight_file(mat['Xtest'], np.reshape(mat['Ytest'],(-1)), 'shuttle_test')