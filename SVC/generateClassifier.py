from sklearn.svm import LinearSVC
from sklearn import preprocessing
from skimage.feature import hog
from sklearn.externals import joblib
import numpy as np
from collections import Counter
import scipy.io


#LOAD ML DATA FROM .MAT FILE
dataset = scipy.io.loadmat('mnist-original.mat')
data = dataset["data"].T
_Labels = dataset["label"].T
features = np.asanyarray(data,'int16')
labels = np.asanyarray(_Labels,'int')


list_hog_fd = []

for feature in features:
    fd = hog(feature.reshape((28,28)),orientations= 9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise= False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd,'float64')    

pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)


clf = LinearSVC()

clf.fit(hog_features,labels)

joblib.dump((clf, pp), "digits_cls.pkl", compress=3)
