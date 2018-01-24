import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics

#  read the data in from files,
#  assign target values 1 for signal, 0 for background
sigData = np.loadtxt('signal.txt')
nSig = sigData.shape[0]
sigTargets = np.ones(nSig)

bkgData = np.loadtxt('background.txt')
nBkg = bkgData.shape[0]
bkgTargets = np.zeros(nBkg)

# concatenate arrays into data X and targets y
X = np.concatenate((sigData, bkgData), 0)
y = np.concatenate((sigTargets, bkgTargets))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# create object for the NN classifies
clf = MLPClassifier(hidden_layer_sizes=(15,10,5), activation='relu',
                    max_iter=20000, random_state=0)
clf.fit(X_train, y_train)

# evaluate its accuracy using the test data
y_pred = clf.predict(X_test)
print('classification accuracy = ', metrics.accuracy_score(y_test, y_pred))


# create object for the Linear Discretion Analysis classifies
linear_classifier = LinearDiscriminantAnalysis()
linear_classifier.fit(X, y)
y_pred_linear = linear_classifier.predict(X_test)
print('Linear classification accuracy = ', metrics.accuracy_score(y_test, y_pred_linear))

# make histogram of decision function for Linear Discrimination Analysis
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = clf.predict_proba(X_test)[:,1]           # for some classifiers use decision_function
tBkg = tTest[y_test==0]
tSig = tTest[y_test==1]
nBins = 50
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
plt.xlabel('decision function $t$', labelpad=3)
plt.ylabel('$f(t)$', labelpad=3)
n, bins, patches = plt.hist(tSig, bins=bins, normed=1, histtype='step', fill=False, color='dodgerblue')
n, bins, patches = plt.hist(tBkg, bins=bins, normed=1, histtype='step', fill=False, color='red', alpha=0.5)
plt.savefig("decision_function__NN_hist.pdf", format='pdf')

plt.show()


# make histogram of decision function for NN
plt.figure()                                     # new window
matplotlib.rcParams.update({'font.size':14})     # set all font sizes
tTest = linear_classifier.predict_proba(X_test)[:,1] # for some classifiers use decision_function
tBkg = tTest[y_test==0]
tSig = tTest[y_test==1]
nBins = 50
tMin = np.floor(np.min(tTest))
tMax = np.ceil(np.max(tTest))
bins = np.linspace(tMin, tMax, nBins+1)
plt.xlabel('decision function $t$', labelpad=3)
plt.ylabel('$f(t)$', labelpad=3)
n, bins, patches = plt.hist(tSig, bins=bins, normed=1, histtype='step', fill=False, color='dodgerblue')
n, bins, patches = plt.hist(tBkg, bins=bins, normed=1, histtype='step', fill=False, color='red', alpha=0.5)
plt.savefig("decision_function_Linear_hist.pdf", format='pdf')

plt.show()
