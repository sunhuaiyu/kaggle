import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('train.csv')

X_train = train.ix[:30000, 1:]
y_train = train.ix[:30000, 0]
X_test = train.ix[30000:40000, 1:]
y_test = train.ix[30000:40000, 0]

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# neural network
f = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                  hidden_layer_sizes=(200, 20), random_state=1)
f.fit(X_train, y_train)
f.score(X_test, y_test)
# 0.9687
# note: hidden_layer_sizes=(1000, 50) didn't increase performance; 
# note: (100, 10) slightly reduced accuracy.

# RandomForest
f = RandomForestClassifier(n_estimators=1000, n_jobs=12)
f.fit(X_train, y_train)
f.score(X_test, y_test)
# 0.966 

# neural network 2nd try
f = MLPClassifier(solver='adam', alpha=1e-5, max_iter=400,
                  hidden_layer_sizes=(200, 100), random_state=1)
f.fit(X_train, y_train)
f.score(X_test, y_test)
# 0.9733, but final submission is 0.96757 not as good as the first one.

# for submission
# normalize data to [0, 1] by dividing training data by 255.0 improved score
X = train.ix[:, 1:] / 255.
y = train.ix[:, 0]
A = pd.read_csv('test.csv') / 255. 

f = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(200, 100), 
                  solver='lbfgs', random_state=1)
f.fit(X, y)
p = f.predict(A)

m = pd.read_csv('sample_submission.csv')
m.Label = p
m.to_csv('sample_submission.csv', index=False)
# rank#759 score=0.96857 on 12/7/16
# rank#1 score=1.00 if I use the entire MNIST data as training set! 12/8/16
# would rand#593  with score=0.97514, by normalizing training data to [0,1].



'''
lb = preprocessing.LabelBinarizer()
lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = lb.transform(y)
'''
