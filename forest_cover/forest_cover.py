from pylab import *
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv', index_col='Id')
X = minmax_scale(train.ix[:, :-1])
y = array(train.ix[:, -1])

test = pd.read_csv('test.csv', index_col='Id')
A = minmax_scale(test)

f = RandomForestClassifier(n_estimators=1000, n_jobs=12)
f.fit(X, y)
p = f.predict(A)

s = pd.read_csv('sampleSubmission.csv', index_col='Id')
s.Cover_Type = p
s.to_csv('sampleSubmission.csv')
# score=0.74480 using RandomForestClassifier;  0.58160 2yrs ago
# scaling to [0,1] made some difference

# NN
from sklearn.neural_network import MLPClassifier

f = MLPClassifier(hidden_layer_sizes=(80, 20), alpha=1e-5, random_state=1)
f.fit(X, y)
p = f.predict(A)

s = pd.read_csv('sampleSubmission.csv', index_col='Id')
s.Cover_Type = p
s.to_csv('sampleSubmission.csv')
# score=0.64248 using a NN with 2 hidden layers, same data scaling
# score=0.63421 if change the activation function to tanh

# activation function: "rectified linear unit function"
h1 = clip(dot(A, f.coefs_[0]) + f.intercepts_[0], 0, a_max=None)
h2 = clip(dot(h1, f.coefs_[1]) + f.intercepts_[1], 0, a_max=None)
out = clip(dot(h2, f.coefs_[2]) + f.intercepts_[2], 0, a_max=None)
out = out.argmax(axis=1) + 1

# activation function: tanh
h1 = tanh(dot(A, f.coefs_[0]) + f.intercepts_[0])
h2 = tanh(dot(h1, f.coefs_[1]) + f.intercepts_[1])
out = tanh(dot(h2, f.coefs_[2]) + f.intercepts_[2])
out = out.argmax(axis=1) + 1


